
import pandapower as pp
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import os
import matplotlib.pyplot as plt

def get_solution(model, data):
    """Get optimal setpoints for the feedforward controller"""

    net = data["net"]
    baseMVA = data["baseMVA"]

    opt_values = {b: {"V": pyo.value(model.V[b]),
                    "theta": pyo.value(model.theta[b]),
                    "Pg": pyo.value(model.Pg[b]),
                    "Qg": pyo.value(model.Qg[b])} for b in model.Buses}

    optimal_setpoints = {}
    slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
    optimal_setpoints["p_mw_slack"] = opt_values[slack_bus]["Pg"] * baseMVA

    # Update other generators.
    optimal_setpoints["p_mw"] = {}
    optimal_setpoints["q_mvar"] = {}
    i = 0
    for idx, row in net.sgen.iterrows():
        bus = int(row["bus"]) 
        if bus in opt_values:
            optimal_setpoints["p_mw"][i] = opt_values[bus]["Pg"] * baseMVA
            optimal_setpoints["q_mvar"][i] = opt_values[bus]["Qg"] * baseMVA
            i += 1

    print(optimal_setpoints)
    
    # === Save compatible CSVs ===
    result_dir = os.path.join(os.path.dirname(__file__), "Results_ipopt")
    os.makedirs(result_dir, exist_ok=True)

    pd.DataFrame(optimal_setpoints["p_mw"], index=[0]).rename(
        columns=lambda x: f"gen_{x}"
    ).to_csv(os.path.join(result_dir, "p_mw.csv"), sep=";", index=False)

    pd.DataFrame(optimal_setpoints["q_mvar"], index=[0]).rename(
        columns=lambda x: f"gen_{x}"
    ).to_csv(os.path.join(result_dir, "q_mvar.csv"), sep=";", index=False)

    pd.DataFrame(optimal_setpoints["V"], index=[0]).rename(
        columns=lambda x: f"bus_{x}"
    ).to_csv(os.path.join(result_dir, "v_bus.csv"), sep=";", index=False)


    return optimal_setpoints

def process_results(model, data):
    """Process the results from solving the pyomo model with IPOPT
    Inputs: model (pyomo_model), data (extract_data)"""

    # Unpack necessary items from the data dictionary.
    net = data["net"]
    baseMVA = data["baseMVA"]
    buses = data["buses"]
    P_load = data["P_load"]
    Q_load = data["Q_load"]
    branch = data["branch"]
    G_matrix = data["G_matrix"]
    B_matrix = data["B_matrix"]
    gen_buses = data["gen_buses"]
    slack_bus = data["slack_bus"]

    # Display the optimization results.
    print("\nOptimized OPF Results with IPOPT:")
    for b in model.Buses:
        V_val = pyo.value(model.V[b])
        theta_val = pyo.value(model.theta[b]) 
        Pg_val = pyo.value(model.Pg[b]) * baseMVA
        Qg_val = pyo.value(model.Qg[b]) * baseMVA
        print(f"Bus {b}: V = {V_val:.3f} pu, theta = {theta_val:.3f} rad, Pg = {Pg_val:.3f} MW, Qg = {Qg_val:.3f} MVAr")

    # Update the pandapower network with the optimized generator set-points
    opt_values = {b: {"V": pyo.value(model.V[b]),
                    "theta": pyo.value(model.theta[b]),
                    "Pg": pyo.value(model.Pg[b]),
                    "Qg": pyo.value(model.Qg[b])} for b in model.Buses}

    # Update ext_grid for the slack bus.
    if not net.ext_grid.empty:
        slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
        net.ext_grid.at[net.ext_grid.index[0], "p_mw"] = opt_values[slack_bus]["Pg"] * baseMVA
        #if "q_mvar" in net.ext_grid.columns:
            #net.ext_grid.at[net.ext_grid.index[0], "q_mvar"] = opt_values[slack_bus]["Qg"] * baseMVA

    # Update other generators.
    for idx, row in net.sgen.iterrows():
        bus = int(row["bus"]) 
        if bus in opt_values:
            net.sgen.at[idx, "p_mw"] = opt_values[bus]["Pg"] * baseMVA
            net.sgen.at[idx, 'q_mvar'] = opt_values[bus]["Qg"] * baseMVA

    # Run the pandapower power flow.
    pp.runpp(net)

    # -----------------------------
    # Comparison
    # -----------------------------

    # Convert pandapower angles from degrees to radians for comparison
    pp_results = net.res_bus.copy()
    pp_results['va_radians'] = np.deg2rad(pp_results['va_degree'])

    # Merge results on bus index
    comparison = pd.DataFrame({
        'V_IPOPT': [opt_values[b]["V"] for b in sorted(opt_values.keys())],
        'V_PP': pp_results['vm_pu'].values,
        'theta_IPOPT': [opt_values[b]["theta"] for b in sorted(opt_values.keys())],
        'theta_PP': pp_results['va_radians'].values
    })

    # Compute differences
    comparison['V_diff'] = comparison['V_IPOPT'] - comparison['V_PP']
    comparison['theta_diff'] = comparison['theta_IPOPT'] - comparison['theta_PP']

    print("\nComparison of Voltages:")
    print(comparison)

    # -----------------------------
    # IPOPT results DataFrame
    # -----------------------------
    ipopt_results = []
    for b in model.Buses:
        Pg_ipopt = pyo.value(model.Pg[b]) * baseMVA  # in MW
        Qg_ipopt = pyo.value(model.Qg[b]) * baseMVA  # in MVAr
        # Compute net injection at bus b (generator output minus load)
        net_inj_P_ipopt = Pg_ipopt - (P_load[b] * baseMVA)
        net_inj_Q_ipopt = Qg_ipopt - (Q_load[b] * baseMVA)
        
        ipopt_results.append({
            "bus": b,
            "Pg_IPOPT": Pg_ipopt,
            "Qg_IPOPT": Qg_ipopt,
            "net_inj_P_IPOPT": net_inj_P_ipopt,
            "net_inj_Q_IPOPT": net_inj_Q_ipopt
        })
    ipopt_df = pd.DataFrame(ipopt_results).set_index("bus")
    pd.set_option('display.float_format', '{:.3f}'.format)

    print("\nIPOPT Results:")
    print(ipopt_df)


    # -----------------------------
    # Build the pandapower manual calculation DataFrame
    # -----------------------------

    # List all buses in the network.
    buses = net.bus.index.tolist()

    # For regular generators: attach the bus info from net.sgen.
    res_sgen = net.res_sgen.copy()
    res_sgen['bus'] = net.sgen['bus']
    grouped_gen_active = res_sgen.groupby('bus')['p_mw'].sum()
    grouped_gen_reactive = res_sgen.groupby('bus')['q_mvar'].sum()

    # For external grids (slack), we use the bus info from net.ext_grid.
    if not net.res_ext_grid.empty:
        res_ext_grid = net.res_ext_grid.copy()
        res_ext_grid['bus'] = net.ext_grid['bus']
        grouped_ext_active = res_ext_grid.groupby('bus')['p_mw'].sum()
        grouped_ext_reactive = res_ext_grid.groupby('bus')['q_mvar'].sum()
    else:
        grouped_ext_active = pd.Series(0, index=buses)
        grouped_ext_reactive = pd.Series(0, index=buses)

    # Total generation on each bus:
    total_gen_active = pd.Series(0, index=buses).add(grouped_gen_active, fill_value=0) + \
                        pd.Series(0, index=buses).add(grouped_ext_active, fill_value=0)
    total_gen_reactive = pd.Series(0, index=buses).add(grouped_gen_reactive, fill_value=0) + \
                        pd.Series(0, index=buses).add(grouped_ext_reactive, fill_value=0)

    # Loads (active and reactive) from net.load (given as positive numbers):
    if not net.load.empty:
        load_active = net.load.groupby('bus')['p_mw'].sum()
        load_reactive = net.load.groupby('bus')['q_mvar'].sum()
    else:
        load_active = pd.Series(0, index=buses)
        load_reactive = pd.Series(0, index=buses)
    load_active = pd.Series(0, index=buses).add(load_active, fill_value=0)
    load_reactive = pd.Series(0, index=buses).add(load_reactive, fill_value=0)

    # Calculate pandapower net injections: Generation minus Load.
    net_inj_P_PP = total_gen_active - load_active
    net_inj_Q_PP = total_gen_reactive - load_reactive

    # Build a DataFrame for pandapower results.
    pp_manual_df = pd.DataFrame({
        "Pg_PP": total_gen_active,
        "Qg_PP": total_gen_reactive,
        "net_inj_P_PP": net_inj_P_PP,
        "net_inj_Q_PP": net_inj_Q_PP
    }, index=buses)

    print("\nPandapower (PP) Results:")
    print(pp_manual_df)


    # -----------------------------
    # Compute Differences
    # -----------------------------

    # Create a DataFrame for differences.
    diff_df = pd.DataFrame({
        "Pg_diff": ipopt_df["Pg_IPOPT"] - pp_manual_df["Pg_PP"],
        "Qg_diff": ipopt_df["Qg_IPOPT"] - pp_manual_df["Qg_PP"],
        "net_inj_P_diff": ipopt_df["net_inj_P_IPOPT"] - pp_manual_df["net_inj_P_PP"],
        "net_inj_Q_diff": ipopt_df["net_inj_Q_IPOPT"] - pp_manual_df["net_inj_Q_PP"]
    }, index=buses)

    print("\nDifferences (IPOPT - Pandapower):")
    print(diff_df)

    print(f"\nFinal IPOPT cost: {pyo.value(model.obj):.4f}")

    # Save IPOPT results and generate plots
    results_dir = "Results_ipopt"
    os.makedirs(results_dir, exist_ok=True)

    # Extract bus-sorted results
    buses_sorted = sorted(opt_values.keys())
    Pg_arr = np.array([opt_values[b]["Pg"] for b in buses_sorted])
    Qg_arr = np.array([opt_values[b]["Qg"] for b in buses_sorted])
    V_arr = np.array([opt_values[b]["V"] for b in buses_sorted])
    theta_arr = np.array([opt_values[b]["theta"] for b in buses_sorted])

   # Format and save results as consistent CSV files
    Pg_df = pd.DataFrame([Pg_arr], columns=[f"P_G{i+1}" for i in range(len(Pg_arr))])
    Qg_df = pd.DataFrame([Qg_arr], columns=[f"Q_G{i+1}" for i in range(len(Qg_arr))])
    V_df  = pd.DataFrame([V_arr],  columns=[f"V_Bus{i+1}" for i in range(len(V_arr))])
    theta_df = pd.DataFrame([theta_arr], columns=[f"Theta_Bus{i+1}" for i in range(len(theta_arr))])

    Pg_df.to_csv(os.path.join(results_dir, "p_mw.csv"), sep=";", index=False)
    Qg_df.to_csv(os.path.join(results_dir, "q_mvar.csv"), sep=";", index=False)
    V_df.to_csv(os.path.join(results_dir, "v_bus.csv"), sep=";", index=False)
    theta_df.to_csv(os.path.join(results_dir, "theta_bus.csv"), sep=";", index=False)

    # Generate and save plots
    print_bus_results(Pg_arr, Qg_arr, V_arr, theta_arr)
    plot_single_bar(Pg_arr, "Active Power Output (Pg)", "Pg (p.u.)", f"{results_dir}/Pg_bar.png", buses_sorted)
    plot_single_bar(Qg_arr, "Reactive Power Output (Qg)", "Qg (p.u.)", f"{results_dir}/Qg_bar.png", buses_sorted)
    plot_single_bar(V_arr, "Voltage Magnitude (V)", "V (p.u.)", f"{results_dir}/V_bar.png", buses_sorted)
    plot_single_bar(np.degrees(theta_arr), "Voltage Angle (θ)", "Angle (deg)", f"{results_dir}/theta_bar.png", buses_sorted)
    plot_combined_subplot(Pg_arr, Qg_arr, V_arr, theta_arr, buses_sorted, f"{results_dir}/opf_combined_subplot.png")

    print(f"\nAll result arrays and plots saved to: '{results_dir}'")

def print_bus_results(Pg, Qg, V, theta):
    """Print IPOPT results for each bus with formatting"""
    print("\n=== IPOPT Results ===")
    print(f"{'Bus':>5} {'Pg (p.u.)':>12} {'Qg (p.u.)':>12} {'V (p.u.)':>12} {'θ (deg)':>12}")
    for i in range(len(Pg)):
        print(f"{i:5} {Pg[i]:12.4f} {Qg[i]:12.4f} {V[i]:12.4f} {np.degrees(theta[i]):12.4f}")

def plot_single_bar(data, title, ylabel, filename, buses):
    """Create and save a single bar chart with labels"""
    plt.figure(figsize=(6, 4))
    bars = plt.bar(buses, data, color='skyblue')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        offset = 0.01 if height >= 0 else -0.03
        plt.text(bar.get_x() + bar.get_width()/2, height + offset,
                 f"{height:.4f}", ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    ymin = min(0, min(data))
    ymax = max(0, max(data))
    if ymax == ymin:
        ymax = ymin + 1
    plt.ylim(ymin * 1.2, ymax * 1.2)
    plt.title(title)
    plt.xlabel("Bus")
    plt.ylabel(ylabel)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    plt.close()

def plot_combined_subplot(Pg, Qg, V, theta, buses, filename):
    """Create and save a 2x2 subplot for Pg, Qg, V, theta"""
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    axs = axs.flatten()
    variables = [Pg, Qg, V, np.degrees(theta)]
    titles = ["Active Power Output (Pg)", "Reactive Power Output (Qg)", "Voltage Magnitude (V)", "Voltage Angle (θ)"]
    ylabels = ["Pg (p.u.)", "Qg (p.u.)", "V (p.u.)", "Angle (deg)"]

    for i in range(4):
        data = variables[i]
        bars = axs[i].bar(buses, data, color='lightcoral')
        for j, bar in enumerate(bars):
            height = bar.get_height()
            offset = 0.01 if height >= 0 else -0.03
            axs[i].text(bar.get_x() + bar.get_width()/2, height + offset,
                        f"{height:.4f}", ha='center', va='bottom' if height >= 0 else 'top', fontsize=7)
        ymin = min(0, min(data))
        ymax = max(0, max(data))
        if ymax == ymin:
            ymax = ymin + 1
        axs[i].set_ylim(ymin * 1.2, ymax * 1.2)
        axs[i].set_title(titles[i], fontsize=10)
        axs[i].set_ylabel(ylabels[i])
        axs[i].set_xlabel("Bus")
        axs[i].grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.suptitle("OPF Solution Overview", fontsize=14)
    plt.savefig(filename)
    plt.close()

