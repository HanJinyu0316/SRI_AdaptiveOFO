import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.networks as pn
from cost_function import compute_cost_function
from grad_cost_function import compute_cost_gradient
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

def main_adaptive(T=3000, alpha0=1e-5, Simulation=True):
    results_dir = "Results_adaptive"
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

        net.sgen["min_p_mw"] = np.array([0, 0], dtype=float)
        net.sgen["max_p_mw"] = np.array([200, 150], dtype=float)
        net.sgen["p_mw"] = np.array([150, 100], dtype=float)
        net.sgen["min_q_mvar"] = np.array([-60, -30], dtype=float)
        net.sgen["max_q_mvar"] = np.array([100, 80], dtype=float)
        net.sgen["q_mvar"] = np.array([0, 0], dtype=float)
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
        umin = np.concatenate((net.sgen.min_p_mw.values, net.sgen.min_q_mvar.values)).astype(float) / baseMVA
        umax = np.concatenate((net.sgen.max_p_mw.values, net.sgen.max_q_mvar.values)).astype(float) / baseMVA
        u = np.clip(u, umin, umax)

        cost_list, p_list, q_list, v_list, line_list = [], [], [], [], []
        alpha = alpha0
        theta_prev = 1.0 / 3
        alpha_list = []
        grad_prev = None
        u_prev = None

        n = len(net.sgen)
        net.sgen["p_mw"] = u[:n] * baseMVA
        net.sgen["q_mvar"] = u[n:] * baseMVA
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            print("[ERROR] Initial power flow did not converge. Check initial conditions.")
            return

        for k in range(T):
            n = len(net.sgen)
            net.sgen["p_mw"] = u[:n] * baseMVA
            net.sgen["q_mvar"] = u[n:] * baseMVA
            
            try:
                pp.runpp(net)
                grad = compute_cost_gradient(net, u, umin, umax, barrier_alpha=100)
            except pp.LoadflowNotConverged:
                print(f"[WARNING] Power flow did not converge at step {k}. Logging NaNs and holding u.")
                alpha_list.append(alpha)
                cost_list.append(np.nan)
                p_list.append(net.sgen["p_mw"].values.copy())
                q_list.append(net.sgen["q_mvar"].values.copy())
                v_list.append(np.full(len(net.bus), np.nan))
                line_list.append(np.full(len(net.line), np.nan))
                continue

            if grad_prev is not None:
                u_diff_norm = max(np.linalg.norm(u - u_prev), 1e-8)
                grad_diff_norm = np.linalg.norm(grad - grad_prev)
                Lk = grad_diff_norm / u_diff_norm
                
                alpha_candidate1 = np.sqrt(2 / 3 + theta_prev) * alpha
                denom = max(2 * alpha**2 * Lk**2, 1e-8)
                alpha_candidate2 = alpha / np.sqrt(denom)
                alpha_new = min(alpha_candidate1, alpha_candidate2)
                
                alpha_min = 1e-6
                alpha_max = 1e-2
                alpha_new = np.clip(alpha_new, alpha_min, alpha_max)
            else:
                alpha_new = alpha

            u_new = prox_operator(u - alpha_new * grad, umin, umax)

            n = len(net.sgen)
            net.sgen["p_mw"] = u_new[:n] * baseMVA
            net.sgen["q_mvar"] = u_new[n:] * baseMVA

            try:
                pp.runpp(net)
            except pp.LoadflowNotConverged:
                print(f"[WARNING] Power flow did not converge after update at step {k}. Logging NaNs and holding u.")
                alpha_list.append(alpha_new)
                cost_list.append(np.nan)
                p_list.append(net.sgen["p_mw"].values.copy())
                q_list.append(net.sgen["q_mvar"].values.copy())
                v_list.append(np.full(len(net.bus), np.nan))
                line_list.append(np.full(len(net.line), np.nan))
                continue

            cost_new = compute_cost_function(net, u_new, umin, umax, barrier_alpha=100)

            alpha_list.append(alpha_new)
            cost_list.append(cost_new)
            p_list.append(net.sgen["p_mw"].values.copy())
            q_list.append(net.sgen["q_mvar"].values.copy())
            v_list.append(net.res_bus["vm_pu"].values.copy())
            line_list.append(net.res_line["loading_percent"].values.copy())

            grad_prev = grad
            u_prev = u
            u = u_new
            theta_prev = alpha_new / alpha
            alpha = alpha_new

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-4:
                print(f"\n[Converged] Step {k}, Cost = {cost_new:.4f}, α = {alpha:.2e}, ||grad|| = {grad_norm:.2e}")
                print_bus_results(net)
                break

            if k % 500 == 0:
                print(f"\n[Adaptive] Step {k}, Cost = {cost_new:.4f}, α = {alpha:.2e},||grad|| = {grad_norm:.2e}")
                print_bus_results(net)

        # Save results
        pd.DataFrame({"cost": cost_list}).to_csv(os.path.join(results_dir, "cost_history.csv"), sep=";", index=False)
        pd.DataFrame(p_list, columns=[f"P_G{i+1}" for i in range(len(p_list[0]))]).to_csv(os.path.join(results_dir, "p_mw.csv"), sep=";", index=False)
        pd.DataFrame(q_list, columns=[f"Q_G{i+1}" for i in range(len(q_list[0]))]).to_csv(os.path.join(results_dir, "q_mvar.csv"), sep=";", index=False)

        v_cols = [f"V_Bus{i}" for i in range(1, v_list[0].shape[0] + 1)]
        pd.DataFrame(v_list, columns=v_cols).to_csv(os.path.join(results_dir, "v_bus.csv"), sep=";", index=False)

        l_cols = [f"Line_{i}" for i in range(1, line_list[0].shape[0] + 1)]
        pd.DataFrame(line_list, columns=l_cols).to_csv(os.path.join(results_dir, "line_loading.csv"), sep=";", index=False)

        pd.DataFrame({'alpha': alpha_list}).to_csv(os.path.join(results_dir, 'alpha_history.csv'), sep=';', index=False)

        plt.figure(figsize=(8, 5))
        plt.plot(alpha_list, label='Step Size α')
        plt.xlabel('Iteration')
        plt.ylabel('Alpha')
        plt.title('Step Size Evolution')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'alpha_evolution.png'))
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(cost_list, label="Cost")
        plt.xlabel("Iteration")
        plt.ylabel("Total Cost")
        plt.title("Cost Convergence (Adaptive Step Size)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "cost_convergence.png"))
        plt.show()

        print("\nFinal solution at step T:")
        for i, bus in enumerate(net.bus.index):
            pg = net.res_ext_grid["p_mw"].sum() if i == 0 else net.res_sgen["p_mw"][net.sgen["bus"] == i].sum()
            qg = net.res_ext_grid["q_mvar"].sum() if i == 0 else net.res_sgen["q_mvar"][net.sgen["bus"] == i].sum()
            v = net.res_bus.at[i, "vm_pu"]
            theta = net.res_bus.at[i, "va_degree"]
            print(f"{i:>3} | {pg:>8.3f} | {qg:>8.3f} | {v:>8.3f} | {theta:>7.2f}")

        print(f"\nAdaptive optimization completed. Results saved to {results_dir}/")

if __name__ == "__main__":
    main_adaptive(Simulation=True)