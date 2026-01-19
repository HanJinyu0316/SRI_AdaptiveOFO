import numpy as np
import pandapower as pp

def compute_sensitivity_matrix(net):
    """
    Computes:
    - S_v: sensitivity of PQ bus voltages to sgen control input u = [P, Q]
           using Jacobian-based linearization
    - dp_du: sensitivity of slack bus active power to u via finite difference

    Returns:
        S_v : ndarray, shape (m, 2n)
        dp_du : ndarray, shape (2n,)
    """
    # Base power-flow at the current operating point.
    pp.runpp(net)
    n_bus = len(net.bus)
    sgen_idx = net.sgen.index
    sgen_list = list(sgen_idx)  # Ensure fixed ordering
    n = len(sgen_list)
    pq_buses = net.bus.index[net.bus["type"] == 1]
    m = len(pq_buses)

    # === Jacobian-based voltage sensitivity ===
    V = net.res_bus.vm_pu.values
    TH = np.deg2rad(net.res_bus.va_degree.values)
    t0 = V * np.exp(1j * TH)

    Yb = net._ppc['internal']['Ybus'].toarray()
    DYt = np.diagflat(np.conj(Yb @ t0))
    Dt  = np.diagflat(t0)

    def bracket(X): return np.block([[X.real, -X.imag], [X.imag, X.real]])
    def Rmatrix(V, TH):
        VV = np.diagflat(V)
        COST = np.diagflat(np.cos(TH))
        SIN  = np.diagflat(np.sin(TH))
        return np.block([[COST, -VV @ SIN], [SIN, VV @ COST]])
    def Nmatrix(i): return np.block([[np.eye(i), np.zeros((i, i))],
                                     [np.zeros((i, i)), -np.eye(i)]])

    Y_R = bracket(Yb)
    DYt_b = bracket(DYt)
    Dt_b  = bracket(Dt)
    N     = Nmatrix(n_bus)
    Rbr   = Rmatrix(V.reshape(-1, 1), TH.reshape(-1, 1))

    Agrid = DYt_b + Dt_b @ N @ Y_R
    A_sys = np.hstack([Agrid @ Rbr, -np.eye(2 * n_bus)])

    def v(i): return i
    def th(i): return n_bus + i
    def p(i): return 2 * n_bus + i
    def q(i): return 3 * n_bus + i

    gen = set(net.gen.loc[net.gen.in_service, 'bus'].values)
    sgen = set(net.sgen.loc[net.sgen.in_service, 'bus'].values)
    loads = set(net.load['bus'].values)
    slack = next(iter(net.ext_grid.loc[net.ext_grid.in_service, 'bus'].values))
    all_b = set(range(n_bus))
    others = all_b - gen - sgen - loads - {slack}

    u_idx, d_idx, s_idx, y_idx = [], [], [], []
    for i in sorted(gen | sgen):
        u_idx += [p(i), q(i)]
    for i in sorted(loads | others):
        d_idx += [p(i), q(i)]
    s_idx += [v(slack), th(slack)]
    y_idx += [v(i) for i in pq_buses]

    z_idx = np.unique(np.concatenate([u_idx, d_idx, s_idx]))
    A_z = A_sys[:, z_idx]
    A_y = A_sys[:, y_idx]
    A_u = A_z[:, :len(np.unique(u_idx))]

    try:
        S_bus = -np.linalg.solve(A_y, A_u)
    except np.linalg.LinAlgError:
        S_bus = -np.linalg.pinv(A_y) @ A_u

    S_v = S_bus  # Voltage magnitude sensitivity (m × 2n)

    # === Slack power sensitivity using finite difference ===
    baseMVA = 100
    base_p_slack = float(net.res_ext_grid.p_mw.values[0])
    dp_du = np.zeros(2 * n)
    epsilon = 1e-5 * baseMVA

    for j in range(2 * n):
        # Save original values
        orig_p = net.sgen["p_mw"].copy()
        orig_q = net.sgen["q_mvar"].copy()

        idx = j % n
        is_p = j < n
        gen_bus = sgen_list[idx]

        # Perturb one variable (ensure float dtype to avoid warnings)
        if is_p:
            # Ensure column is float type before assignment
            if net.sgen["p_mw"].dtype != 'float64':
                net.sgen["p_mw"] = net.sgen["p_mw"].astype('float64')
            net.sgen.at[gen_bus, "p_mw"] = float(net.sgen.at[gen_bus, "p_mw"]) + epsilon
        else:
            # Ensure column is float type before assignment
            if net.sgen["q_mvar"].dtype != 'float64':
                net.sgen["q_mvar"] = net.sgen["q_mvar"].astype('float64')
            net.sgen.at[gen_bus, "q_mvar"] = float(net.sgen.at[gen_bus, "q_mvar"]) + epsilon

        try:
            pp.runpp(net)
            pert_p_slack = float(net.res_ext_grid.p_mw.values[0])
            dp_du[j] = (pert_p_slack - base_p_slack) / epsilon
        except pp.LoadflowNotConverged:
            dp_du[j] = 0.0  # fallback in case of divergence

        # Restore original values
        net.sgen["p_mw"] = orig_p
        net.sgen["q_mvar"] = orig_q

    # IMPORTANT: after the finite-difference loop, the last successful PF was
    # evaluated at a *perturbed* operating point. Re-run PF at the restored
    # (base) setpoints so that downstream code sees consistent results.
    pp.runpp(net)

    dp_du /= baseMVA
    return S_v, dp_du
