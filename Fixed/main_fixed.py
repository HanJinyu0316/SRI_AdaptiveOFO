import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
from grad_cost_function import compute_cost_gradient
from cost_function import compute_cost_function
from projection import prox_operator

def print_bus_results(net):
    print("Bus |    Pg    |    Qg    |    V     |  Theta")
    print("----+----------+----------+----------+---------")
    for i, bus in enumerate(net.bus.index):
        pg = 0.0
        qg = 0.0
        if bus in net.res_ext_grid.index:
            pg += net.res_ext_grid.at[bus, "p_mw"]
            qg += net.res_ext_grid.at[bus, "q_mvar"]
        if bus in net.sgen["bus"].values:
            for idx in net.sgen[net.sgen["bus"] == bus].index:
                pg += net.res_sgen.at[idx, "p_mw"]
                qg += net.res_sgen.at[idx, "q_mvar"]
        v = net.res_bus.at[bus, "vm_pu"]
        theta = net.res_bus.at[bus, "va_degree"]
        print(f"{i:>3} | {pg:>8.3f} | {qg:>8.3f} | {v:>8.3f} | {theta:>7.2f}")

def main_fixed(T=3000, alpha=1e-5, Simulation=True):
    results_dir = "Results_fixed"
    os.makedirs(results_dir, exist_ok=True)

    if Simulation:
        net = pn.case9()
        for idx, gen in net.gen.iterrows():
            pp.create_sgen(net, bus=gen.bus, p_mw=gen.p_mw, q_mvar=0.0,
                           name=gen.get("name", f"sgen_{idx}"),
                           max_p_mw=gen.max_p_mw, min_p_mw=gen.min_p_mw,
                           max_q_mvar=gen.max_q_mvar, min_q_mvar=gen.min_q_mvar)
        net.gen.drop(net.gen.index, inplace=True)

        baseMVA = 100

        net.sgen["min_p_mw"] = np.array([0, 0]) 
        net.sgen["max_p_mw"] = np.array([200, 150]) 
        net.sgen["p_mw"] = np.array([150, 100])  # start values
        net.sgen["min_q_mvar"] = np.array([-60, -30]) 
        net.sgen["max_q_mvar"] = np.array([100, 80]) 
        net.sgen["q_mvar"] = np.array([0, 0])  # start values
        net.sgen["cost_a"] = np.array([200.0, 100.0]) 
        net.sgen["cost_b"] = np.array([17.5, 7.5]) 
        net.sgen["cost_a_q"] = np.array([100.0, 100.0]) 

        net.ext_grid["cost_a"] = [500.0]
        net.ext_grid["cost_b"] = [50.0]
        net.ext_grid["min_p_mw"] = [0]
        net.ext_grid["max_p_mw"] = [300]
        net.ext_grid["min_q_mvar"] = [-300]
        net.ext_grid["max_q_mvar"] = [300]

        net.load["p_mw"] = np.array([80, 100, 110]) 
        net.load["q_mvar"] = np.array([20, 25, 30]) 

        p_pu = net.sgen["p_mw"].values / baseMVA
        q_pu = net.sgen["q_mvar"].values / baseMVA
        u = np.concatenate([p_pu, q_pu]).astype(float)
        umin = np.concatenate((net.sgen.min_p_mw.values, net.sgen.min_q_mvar.values)).astype(float)/baseMVA
        umax = np.concatenate((net.sgen.max_p_mw.values, net.sgen.max_q_mvar.values)).astype(float)/baseMVA
        u = np.clip(u, umin, umax)

        cost_list, p_list, q_list, v_list, line_list = [], [], [], [], []

        for k in range(T):
            try:
                pp.runpp(net)
            except pp.LoadflowNotConverged:
                print(f"[WARNING] Step {k}: Power flow failed before gradient.")
                continue

            prev_cost = compute_cost_function(net, u, umin, umax, barrier_alpha=100)
            grad_J = compute_cost_gradient(net, u, umin, umax, barrier_alpha=100)
            u_candidate = prox_operator(u - alpha * grad_J, umin, umax)

            # Apply candidate and test
            n = len(net.sgen)
            net.sgen["p_mw"] = u_candidate[:n] * baseMVA
            net.sgen["q_mvar"] = u_candidate[n:] * baseMVA

            try:
                pp.runpp(net)
                new_cost = compute_cost_function(net, u_candidate, umin, umax, barrier_alpha=100)
            except pp.LoadflowNotConverged:
                print(f"[Fallback] Step {k}: Power flow failed after update. Skipping update.")
                continue

            # Accept update
            u = u_candidate
            cost_list.append(new_cost)
            p_list.append(net.sgen["p_mw"].values.copy())
            q_list.append(net.sgen["q_mvar"].values.copy())
            v_list.append(net.res_bus["vm_pu"].values.copy())
            line_list.append(net.res_line["loading_percent"].values.copy())

            if k % 500 == 0:
                print(f"\n[Fixed] Step {k}, Cost = {new_cost:.4f}, α = {alpha:.2e}, ||grad|| = {np.linalg.norm(grad_J):.2e}")
                print_bus_results(net)

            if np.linalg.norm(grad_J) < 1e-4:
                print(f"Early stopping at step {k}, gradient norm = {np.linalg.norm(grad_J):.2e}")
                break

        pd.DataFrame({"cost": cost_list}).to_csv(os.path.join(results_dir, "cost_history.csv"), sep=";", index=False)
        pd.DataFrame(p_list, columns=[f"P_G{i+1}" for i in range(len(p_list[0]))]).to_csv(os.path.join(results_dir, "p_mw.csv"), sep=";", index=False)
        pd.DataFrame(q_list, columns=[f"Q_G{i+1}" for i in range(len(q_list[0]))]).to_csv(os.path.join(results_dir, "q_mvar.csv"), sep=";", index=False)

        v_cols = [f"V_Bus{i}" for i in range(1, v_list[0].shape[0] + 1)]
        pd.DataFrame(v_list, columns=v_cols).to_csv(os.path.join(results_dir, "v_bus.csv"), sep=";", index=False)

        l_cols = [f"Line_{i}" for i in range(1, line_list[0].shape[0] + 1)]
        pd.DataFrame(line_list, columns=l_cols).to_csv(os.path.join(results_dir, "line_loading.csv"), sep=";", index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(cost_list, label="Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost")
        plt.title("Cost Convergence (Fixed Step Size)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "cost_convergence.png"))
        plt.show()

        grad_final = compute_cost_gradient(net, u, umin, umax)
        print("\nFinal gradient vector:\n", grad_final)
        print("\nFinal gradient norm: {:.6f}".format(np.linalg.norm(grad_final)))
        print("\nFinal solution at step T:")
        print_bus_results(net)

        print(f"\nFixed-step optimization completed. Results saved to {results_dir}/")

    else:
        print("\n[INFO] Plotting existing results from Results_fixed...")
        cost = pd.read_csv(os.path.join(results_dir, "cost_history.csv"), sep=";")["cost"].values
        p = pd.read_csv(os.path.join(results_dir, "p_mw.csv"), sep=";").values[-1]
        q = pd.read_csv(os.path.join(results_dir, "q_mvar.csv"), sep=";").values[-1]
        v = pd.read_csv(os.path.join(results_dir, "v_bus.csv"), sep=";").values[-1]

        plt.figure(figsize=(8, 5))
        plt.plot(cost, label="Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost")
        plt.title("Fixed Step Cost Convergence (from saved data)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "cost_convergence_reload.png"))
        plt.show()

        print("\nFinal solution from saved data:")
        theta = np.zeros_like(v)
        print_bus_results(p, q, v, theta)

if __name__ == "__main__":
    main_fixed(Simulation=True)
