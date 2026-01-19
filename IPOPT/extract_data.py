import pandapower as pp
import pandapower.converter as converter
import numpy as np

def extract_data(net):
    """Extracting data the network from pandapower
    Returns necessary data for building the pyomo model"""

    # Convert the pandapower network to a PYPOWER case using a flat start.
    ppc = converter.to_ppc(net, init="flat")
    baseMVA = ppc["baseMVA"]
    bus_ppc = ppc["bus"]
    branch = ppc["branch"]
    
    # makeYbus needs at least 26 columns
    required_cols = 26
    if branch.shape[1] < required_cols:
        missing_cols = required_cols - branch.shape[1]
        branch = np.hstack([branch, np.zeros((branch.shape[0], missing_cols))])

    # Compute Ybus
    Ybus, Yf, Yt = pp.makeYbus_pypower(baseMVA, bus_ppc, branch)
    Ybus = Ybus.todense()  # dense matrix
    G_matrix = np.real(Ybus)
    B_matrix = np.imag(Ybus)

    # Extract bus list and load data.
    buses = net.bus.index.tolist()
    P_load = {b: 0.0 for b in buses}
    Q_load = {b: 0.0 for b in buses}
    for idx, row in net.load.iterrows():
        bus = row["bus"]
        P_load[bus] += row["p_mw"] / baseMVA
        Q_load[bus] += row.get("q_mvar", 0.0) / baseMVA

    # Identify generator buses from net.ext_grid and net.sgen.
    gen_buses = []
    for idx, row in net.sgen.iterrows():
        gen_buses.append(int(row["bus"]))

    slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])

    data = {
        "net": net,
        "baseMVA": baseMVA,
        "buses": buses,
        "P_load": P_load,
        "Q_load": Q_load,
        "branch": branch,
        "G_matrix": G_matrix,
        "B_matrix": B_matrix,
        "gen_buses": gen_buses,
        "slack_bus": slack_bus,
    }

    return data
