import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from extract_data import extract_data
from pyomo_model import build_pyomo_model, solve_model
from grad_cost_function import compute_cost_gradient
from results import process_results, get_solution
import pandapower.networks as pn
import pandapower as pp
import numpy as np

def main():
    # To run one OPF
    net = pn.case9()
    for idx, gen in net.gen.iterrows():
        pp.create_sgen(net, bus=gen.bus, p_mw=gen.p_mw, q_mvar=0, name=gen.get("name", f"sgen_{idx}"),
                   max_p_mw=gen.max_p_mw, min_p_mw=gen.min_p_mw, max_q_mvar=gen.max_q_mvar, min_q_mvar=gen.min_q_mvar)
        
    net.gen.drop(net.gen.index, inplace=True)

    # Generators min/max
    net.sgen["min_p_mw"] = np.array([0, 0])
    net.sgen["max_p_mw"] = np.array([200, 150])
    net.sgen["min_q_mvar"] = np.array([-60, -30])
    net.sgen["max_q_mvar"] = np.array([100, 80])

    # Costs
    net.sgen["cost_a"] = np.array([2e2, 1e2])
    net.sgen["cost_b"] = np.array([1.75e1, 0.75e1])
    net.sgen["cost_a_q"] = np.array([1e2, 1e2])

    # Slack min/max
    net.ext_grid["min_p_mw"] = np.array([0])
    net.ext_grid["max_p_mw"] = np.array([300])
    net.ext_grid["min_q_mvar"] = np.array([-300])
    net.ext_grid["max_q_mvar"] = np.array([300])
    # net.ext_grid["cost_a"] = np.array([5e2])
    net.ext_grid["cost_a"] = np.array([1e4])
    net.ext_grid["cost_b"] = np.array([5e1])

    # Loads
    net.load["p_mw"] = np.array([80, 100, 110])
    net.load["q_mvar"] = np.array([20, 25, 30])

    # Start values
    net.sgen["p_mw"] = np.array([150,100]) 
    net.sgen["q_mvar"] = np.array([-3,-3])

    data = extract_data(net)
    model = build_pyomo_model(data)
        # Run power flow to get initial results
    pp.runpp(net)

    # === Compute and print initial cost (before optimization) ===
    baseMVA = 100.0
    barrier_alpha = 100
    cost = 0.0

    for i in net.sgen.index:
        p_i = net.sgen.at[i, "p_mw"] / baseMVA
        p_ref = net.sgen.at[i, "max_p_mw"] / baseMVA
        dp = p_ref - p_i
        a_i = net.sgen.at[i, "cost_a"]
        b_i = net.sgen.at[i, "cost_b"]
        cost += a_i * dp**2 + b_i * dp

        q_i = net.sgen.at[i, "q_mvar"] / baseMVA
        a_q_i = net.sgen.at[i, "cost_a_q"]
        cost += a_q_i * q_i**2

    p_slack = net.res_ext_grid.p_mw.values[0] / baseMVA
    a_slack = net.ext_grid["cost_a"].values[0]
    b_slack = net.ext_grid["cost_b"].values[0]
    # cost += a_slack * p_slack**2 + b_slack * p_slack
    cost += a_slack * p_slack**2 

    if p_slack < 0:
        cost += 1e4 * abs(p_slack)
    
    if p_slack > 1:
        total_cost += 1e4 * p_slack

    V = net.res_bus.vm_pu[net.bus["type"] == 1].values
    barrier = -np.sum(np.log(1.1 - V)) - np.sum(np.log(V - 0.9))
    cost += barrier_alpha * barrier

    print(f"\n[Step 1] Initial Cost (before optimization): {cost:.4f}\n")

    solve_model(model)
    process_results(model, data) # optimal solution and comparison with pandapower

def feedforward_opf(net):
    data = extract_data(net)
    model = build_pyomo_model(data)
    solve_model(model)
    optimal_setpoints = get_solution(model, data)
    return optimal_setpoints

if __name__ == "__main__":
    main()    