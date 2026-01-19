import numpy as np
import pandapower as pp

def compute_cost_function(net, u_val, umin=None, umax=None, barrier_alpha=1e-2):
    """
    Computes the total cost function J(u) with:
    - Curtailment cost
    - Slack power cost
    - Reactive power cost
    - Voltage log barrier penalty (based on V, not u)

    Parameters:
        net : pandapower network
        u_val : ndarray, control input [p1, p2, ..., q1, q2, ...]
        umin, umax : not used for voltage penalty; kept for compatibility
        barrier_alpha : float, penalty strength
    """
    baseMVA = 100
    n = len(net.sgen)
    total_cost = 0.0

    # Curtailment cost (P)
    for i in net.sgen.index:
        p_i = u_val[i]
        p_ref_i = net.sgen.at[i, "max_p_mw"]/baseMVA
        a_i = net.sgen.at[i, "cost_a"] 
        b_i = net.sgen.at[i, "cost_b"] 
        dp = p_ref_i - p_i
        total_cost += a_i * dp**2 + b_i * dp

    # Reactive cost (Q)
    for i in net.sgen.index:
        q_i = u_val[n + i]
        a_q_i = net.sgen.at[i, "cost_a_q"] 
        total_cost += a_q_i * q_i**2

    # Slack bus cost
    p_slack = float(net.res_ext_grid["p_mw"].values[0])/baseMVA
    # penalty_strength = 1e4
    # total_cost += penalty_strength * p_slack**2

    a_slack = float(net.ext_grid["cost_a"].values[0]) 
    b_slack = float(net.ext_grid["cost_b"].values[0])
    total_cost += a_slack * p_slack**2 + b_slack * p_slack

    # total_cost += 1e4 * (1/(1 + np.exp(p_slack)))

    if p_slack < 0:
        total_cost += 1e4 * abs(p_slack)

    if p_slack > 1:
        total_cost += 1e4 * p_slack

    # # Smoothly penalize p_slack < 0
    # total_cost += 1e4 * np.log1p(np.exp(-p_slack))

    # # Smoothly penalize p_slack > 1
    # total_cost += 1e4 * np.log1p(np.exp(p_slack - 1))


    # if p_slack < 0:
    #      total_cost += 1e4 * np.log1p(np.exp(-p_slack))



    # Voltage log barrier penalty (based on bus voltages)
    V = net.res_bus.loc[net.bus["type"] == 1, "vm_pu"].values
    vmin = np.array([0.9] * len(V))
    vmax = np.array([1.1] * len(V))

    upper_diff = vmax - V
    lower_diff = V - vmin

    if np.any(upper_diff <= 0) or np.any(lower_diff <= 0):
        return np.inf

    barrier = -np.sum(np.log(upper_diff)) - np.sum(np.log(lower_diff))
    total_cost += barrier_alpha * barrier

    return total_cost
