import numpy as np
import pandapower as pp
from sensitivity import compute_sensitivity_matrix

def compute_cost_gradient(net, u_val, umin=None, umax=None, barrier_alpha=1e-2):
    """
    Compute gradient of total cost function:
    - Curtailment cost (P)
    - Reactive power cost (Q)
    - Slack power cost (via chain rule)
    - Voltage barrier penalty (via chain rule and Jacobian-based sensitivity)

    Returns:
        grad : ndarray, shape (2n,)
    """
    n = len(net.sgen)
    baseMVA = 100
    grad = np.zeros_like(u_val)

    # P cost gradient
    for i in net.sgen.index:
        p_i = u_val[i]
        p_ref = net.sgen.at[i, "max_p_mw"]/baseMVA
        a = net.sgen.at[i, "cost_a"] 
        b = net.sgen.at[i, "cost_b"] 
        grad[i] = -2 * a * (p_ref - p_i) - b

    # Q cost gradient
    for i in net.sgen.index:
        q_i = u_val[n + i]
        a_q = net.sgen.at[i, "cost_a_q"] 
        grad[n + i] = 2 * a_q * q_i

    # Get sensitivities (this helper runs a power flow internally).
    # IMPORTANT: avoid running multiple PFs here to keep the gradient evaluation
    # consistent with the sensitivity linearization.
    S_v, dp_du = compute_sensitivity_matrix(net)

    # Slack gradient (chain rule)
    p_slack = float(net.res_ext_grid["p_mw"].values[0])/baseMVA
    a_s = float(net.ext_grid["cost_a"].values[0]) 
    b_s = float(net.ext_grid["cost_b"].values[0]) 
    dJ_dp_slack = 2 * a_s * p_slack + b_s

    # Penalty for p_slack < 0
    if p_slack < 0:
        dJ_dp_slack -= 1e4
    # if p_slack < 0:
    #     dJ_dp_slack += -1e4 / (1 + np.exp(p_slack))
    # # Penalty for p_slack > 1
    if p_slack > 1:
        dJ_dp_slack += 1e4
    # penalty_strength = 1e4
    # dJ_dp_slack = 2 * penalty_strength * p_slack
    
    # dJ_dp_slack = -1e5 * np.exp(p_slack) / (1 + np.exp(p_slack))**2

   
    grad += dJ_dp_slack * dp_du

    # Voltage penalty gradient
    V = net.res_bus.loc[net.bus["type"] == 1, "vm_pu"].values
    vmin = np.array([0.9] * len(V))
    vmax = np.array([1.1] * len(V))
    upper_diff = vmax - V
    lower_diff = V - vmin
    if np.any(upper_diff <= 0) or np.any(lower_diff <= 0):
        return np.full_like(u_val, np.nan)

    dJ_dV = - barrier_alpha * (1.0 / lower_diff - 1.0 / upper_diff)
    grad += S_v.T @ dJ_dV

    return grad
