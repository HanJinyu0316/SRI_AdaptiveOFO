# === Manually specify IPOPT cost value here ===
ipopt_cost_value = 24.5701 # <-- You can modify this value as needed

import os
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def load_csv(folder, name):
    path = os.path.join(folder, name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            try:
                return pd.read_csv(path, sep=",")
            except:
                print(f"[ERROR] Failed to read {path}")
                return None
    return None

def plot_cost(costs, folder):
    plt.figure(figsize=(10, 6))
    for method, cost in costs.items():
        plt.plot(cost, label=method, linestyle=line_styles[method], linewidth=2)
    plt.title("Cost Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Cost")
    _place_legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "cost_comparison.png"))
    plt.close()

def plot_group(data_dict, title, ylabel, filename, folder, scale_factor=1.0):
    plt.figure(figsize=(12, 6))
    reference_df = next((df for df in data_dict.values() if df is not None), None)
    if reference_df is None:
        print(f"[WARNING] Skipping {title} — no data.")
        return
    all_columns = [set(df.columns) for df in data_dict.values() if df is not None]
    columns = sorted(set.intersection(*all_columns)) if all_columns else []
    if not columns:
        print(f"[WARNING] No common columns found for {title}. Skipping plot.")
        return
    color_cycle = cycle(plt.colormaps["tab10"].colors)

    for i, color in zip(range(len(columns)), color_cycle):
        for method in data_dict:
            df = data_dict[method]
            if df is not None and columns[i] in df.columns:
                label = f"{method} - {columns[i]}"
                plt.plot(df[columns[i]] * scale_factor, label=label,
                         linestyle=line_styles.get(method, "-"), color=color, linewidth=2 if ylabel == "V (p.u.)" else 1.5)

    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    _place_legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename))
    plt.close()

def plot_alpha(alpha_data, folder, alpha_fixed=1e-5):
    plt.figure(figsize=(10, 6))
    max_len = max(len(v) for v in alpha_data.values())
    for method in ["Adaptive", "Fixed"]:
        if method == "Fixed":
            plt.plot([alpha_fixed] * max_len, label="Fixed α", linestyle="dashed", linewidth=2)
        elif method in alpha_data:
            plt.plot(alpha_data[method], label="Adaptive α", linestyle="dotted", linewidth=2)
    plt.yscale("log")
    plt.title("Step Size Evolution (log scale)")
    plt.xlabel("Time Step")
    plt.ylabel("Step Size α")
    _place_legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "alpha_comparison_logy.png"))
    plt.close()

def _place_legend():
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               ncol=4, fontsize="small", frameon=False)

def main():
    base = os.path.dirname(__file__)
    folders = {
        "IPOPT": os.path.join(base, "IPOPT", "Results_ipopt"),
        "Fixed": os.path.join(base, "Fixed", "Results_fixed"),
        "Adaptive": os.path.join(base, "Adaptive", "Results_adaptive"),
    }

    out_dir = os.path.join(base, "Comparison")
    os.makedirs(out_dir, exist_ok=True)

    raw = {}
    for method, folder in folders.items():
        raw[method] = {
            "cost": load_csv(folder, "cost_history.csv"),
            "p": load_csv(folder, "p_mw.csv"),
            "q": load_csv(folder, "q_mvar.csv"),
            "v": load_csv(folder, "v_bus.csv"),
            "alpha": load_csv(folder, "alpha_history.csv")
        }

    # Plot cost
    cost_data = {m: raw[m]["cost"]["cost"] for m in raw if raw[m]["cost"] is not None}
    # Inject IPOPT flat cost line if missing
    if "IPOPT" not in cost_data:
        ref_method = "Adaptive" if "Adaptive" in cost_data else "Fixed"
        if ref_method in cost_data:
            T = len(cost_data[ref_method])
            cost_data["IPOPT"] = [ipopt_cost_value] * T

    plot_cost(cost_data, out_dir)

    # Plot P/Q/V with enhanced visuals
    plot_group({m: raw[m]["p"] for m in raw if raw[m]["p"] is not None},
               "Active Power (P) Comparison", "P (MW)", "p_comparison.png", out_dir)
    plot_group({m: raw[m]["q"] for m in raw if raw[m]["q"] is not None},
               "Reactive Power (Q) Comparison", "Q (MVAr)", "q_comparison.png", out_dir)
    plot_group({m: raw[m]["v"] for m in raw if raw[m]["v"] is not None},
               "Voltage Magnitude (V) Comparison", "V (p.u.)", "v_comparison.png", out_dir)

    # Plot α
    alpha_data = {m: raw[m]["alpha"]["alpha"] for m in raw if raw[m]["alpha"] is not None}
    plot_alpha(alpha_data, out_dir)

    print(f"All enhanced plots saved to {out_dir}/")

# Line styles for each method
line_styles = {
    "IPOPT": "solid",
    "Fixed": "dashed",
    "Adaptive": "dotted"
}

if __name__ == "__main__":
    main()