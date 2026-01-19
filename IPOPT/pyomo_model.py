import numpy as np
import pyomo.environ as pyo
from pyomo.environ import log 
from math import radians, cos, sin


def build_pyomo_model(data):
    """Builds the model for which is used for solving OPF
    Returns the model for processing results after solving"""

    # Unpack necessary items from the data dictionary
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

    # Build the cost dictionaries
    cost_a = {}
    cost_b = {}
    cost_a_q = {}
    p_min = {}
    p_max = {}
    q_min = {}
    q_max = {}

    # For the generators from net.sgen:
    for idx, row in net.sgen.iterrows():
        bus = int(row["bus"])
        if "min_p_mw" in row and "max_p_mw" in row:
            p_min[bus] = row["min_p_mw"]
            p_max[bus] = row["max_p_mw"]
            q_min[bus] = row["min_q_mvar"]
            q_max[bus] = row["max_q_mvar"]
            cost_a[bus] = row["cost_a"]
            cost_b[bus] = row["cost_b"]
            cost_a_q[bus] = row["cost_a_q"]
        
    # For slack bus (ext_grid)
    if "min_p_mw" in net.ext_grid.columns:
        slack_bus = int(net.ext_grid.at[net.ext_grid.index[0], "bus"])
        p_min[slack_bus] = 0 #net.ext_grid.at[net.ext_grid.index[0], "min_p_mw"]
        p_max[slack_bus] = net.ext_grid.at[net.ext_grid.index[0], "max_p_mw"]
        q_min[slack_bus] = net.ext_grid.at[net.ext_grid.index[0], "min_q_mvar"]
        q_max[slack_bus] = net.ext_grid.at[net.ext_grid.index[0], "max_q_mvar"]
        cost_a[slack_bus] = net.ext_grid.at[net.ext_grid.index[0], "cost_a"]
        cost_b[slack_bus] = net.ext_grid.at[net.ext_grid.index[0], "cost_b"] 
        cost_a_q[slack_bus] = 0.0


    # Build the Pyomo model.
    model = pyo.ConcreteModel()
    model.Buses = pyo.Set(initialize=buses)

    # ------------- Constraints -------------
    # Voltage magnitude variables
    def V_bounds(model, b):
        return (0.9, 1.1)
    model.V = pyo.Var(model.Buses, domain=pyo.PositiveReals, bounds=V_bounds, initialize=1.0)

    # Voltage angle variables
    def theta_bounds(model, b):
        if b == slack_bus:
            return (0.0, 0.0)
        else:
            return (-0.5, 0.5)  # close to 30 degrees
    model.theta = pyo.Var(model.Buses, domain=pyo.Reals, bounds=theta_bounds, initialize=0.0)

    # AC Branch Flow Constraints
    n_lines = branch.shape[0]
    model.Lines = pyo.RangeSet(0, n_lines-1)

    # Define parameters from branch data
    def init_f_bus(m, i):
        return int(branch[i, 0])
    model.f_bus = pyo.Param(model.Lines, initialize=init_f_bus)

    def init_t_bus(m, i):
        return int(branch[i, 1])
    model.t_bus = pyo.Param(model.Lines, initialize=init_t_bus)

    def init_r(m, i):
        return branch[i, 2]
    model.r = pyo.Param(model.Lines, initialize=init_r)

    def init_x(m, i):
        return branch[i, 3]
    model.x = pyo.Param(model.Lines, initialize=init_x)

    def init_rateA(m, i):
        # Converting the thermal limit (MW) to per unit by dividing by baseMVA
        return branch[i, 5] / baseMVA
    model.rateA = pyo.Param(model.Lines, initialize=init_rateA)

    # Compute the series admittance of each branch.
    # y = 1/(r + j*x) = g + j*b
    def init_g(m, i):
        r_val = model.r[i]
        x_val = model.x[i]
        denom = r_val**2 + x_val**2 
        return r_val / denom if denom != 0 else 0.0
    model.g = pyo.Param(model.Lines, initialize=init_g)

    def init_b(m, i):
        r_val = model.r[i]
        x_val = model.x[i]
        denom = r_val**2 + x_val**2
        return -x_val / denom if denom != 0 else 0.0
    model.b = pyo.Param(model.Lines, initialize=init_b)

    # For line currents
    model.shunt_g = pyo.Param(model.Lines, initialize=lambda m,i: branch[i,4])  
    model.shunt_b = pyo.Param(model.Lines, initialize=lambda m,i: branch[i,5])

    def I2_expr(m, i):
        # from/to buses
        f = m.f_bus[i]
        t = m.t_bus[i]
        # series admittance
        g, b = m.g[i], m.b[i]
        # half‐shunt admittance at the from end
        gsh, bsh = m.shunt_g[i]/2, m.shunt_b[i]/2
        # voltages
        Vf, Vt = m.V[f], m.V[t]
        θf, θt = m.theta[f], m.theta[t]

        # compute ΔV phasor from f→t
        dV_real = Vf*pyo.cos(θf) - Vt*pyo.cos(θt)
        dV_imag = Vf*pyo.sin(θf) - Vt*pyo.sin(θt)

        # series‐branch current I_s = y_ft * ΔV
        I_s_real =  g*dV_real - b*dV_imag
        I_s_imag =  b*dV_real + g*dV_imag

        # shunt current I_sh = (y_sh/2) * V_f e^{j θ_f}
        I_sh_real = gsh*Vf*pyo.cos(θf) - bsh*Vf*pyo.sin(θf)
        I_sh_imag = gsh*Vf*pyo.sin(θf) + bsh*Vf*pyo.cos(θf)

        # total phasor at the from end
        I_tot_real = I_s_real + I_sh_real
        I_tot_imag = I_s_imag + I_sh_imag

        # magnitude squared
        return I_tot_real**2 + I_tot_imag**2

    model.I2 = pyo.Expression(model.Lines, rule=I2_expr)

    model.Imax = pyo.Param(model.Lines, initialize=lambda m, i: 
        branch[i,5]  # if branch[:,5] is the thermal rating in MVA at V=1.0, this is also Imax in p.u.
    )

    # Enforce the current limit
    def current_limit_rule(m, i):
        return m.I2[i] <= m.Imax[i]**2
    model.current_limit = pyo.Constraint(model.Lines, rule=current_limit_rule)

    # ------------- Model Rules -------------
    # Enforce slack bus conditions.
    def slack_voltage_rule(model):
        return model.V[slack_bus] == 1.0
    model.slack_voltage = pyo.Constraint(rule=slack_voltage_rule)

    def slack_angle_rule(model):
        return model.theta[slack_bus] == 0.0
    model.slack_angle = pyo.Constraint(rule=slack_angle_rule)

    # Generator variables for active and reactive power.
    model.Pg = pyo.Var(model.Buses, domain=pyo.Reals, initialize=0.0)
    model.Qg = pyo.Var(model.Buses, domain=pyo.Reals, initialize=0.0)

    # Additional variable: Slack power negative penalty component
    model.Pg_neg = pyo.Var(model.Buses, within=pyo.NonNegativeReals, initialize=0.0)

    # Constraint: Pg_neg[b] >= -Pg[b] for all buses (only active if Pg[b] < 0)
    def slack_pg_penalty_rule(model, b):
        return model.Pg_neg[b] >= -model.Pg[b]
    model.slack_pg_penalty = pyo.Constraint(model.Buses, rule=slack_pg_penalty_rule)

    # Active power balance constraints at each bus.
    def active_power_balance_rule(model, i):
        return model.Pg[i] - P_load[i] == sum(
            model.V[i] * model.V[j] * (
                G_matrix[i, j] * pyo.cos(model.theta[i] - model.theta[j]) +
                B_matrix[i, j] * pyo.sin(model.theta[i] - model.theta[j])
            )
            for j in model.Buses
        )
    model.active_balance = pyo.Constraint(model.Buses, rule=active_power_balance_rule)

    # Reactive power balance constraints at each bus.
    def reactive_power_balance_rule(model, i):
        return model.Qg[i] - Q_load[i] == sum(
            model.V[i] * model.V[j] * (
                G_matrix[i, j] * pyo.sin(model.theta[i] - model.theta[j]) -
                B_matrix[i, j] * pyo.cos(model.theta[i] - model.theta[j])
            )
            for j in model.Buses
        )
    model.reactive_balance = pyo.Constraint(model.Buses, rule=reactive_power_balance_rule)

    def gen_active_limits_rule(model, b):
        if b in p_min:  # i.e., b is slack or a known gen in our dictionary
            return pyo.inequality(p_min[b]/baseMVA, model.Pg[b], p_max[b]/baseMVA)
        else:
            # If not in the table, no generation => Pg=0
            return model.Pg[b] == 0

    model.gen_active_limits = pyo.Constraint(model.Buses, rule=gen_active_limits_rule)

    def gen_reactive_limits_rule(model, b):
        if b in q_min:
            return pyo.inequality(q_min[b]/baseMVA, model.Qg[b], q_max[b]/baseMVA)
        else:
            return model.Qg[b] == 0

    model.gen_reactive_limits = pyo.Constraint(model.Buses, rule=gen_reactive_limits_rule)

    barrier_alpha = 1e-2

    # ------------- Objective Rule -------------
    def objective_rule(model):
        cost_expr = 0.0
        for b in model.Buses:
            if b in cost_a:  # known generator
                a_val = cost_a[b]
                b_val = cost_b[b]
                if b == slack_bus: # slack bus cost
                    # cost_expr += a_val * model.Pg[b]**2 + b_val * model.Pg[b]
                    cost_expr += a_val * model.Pg[b]**2 
                else:
                    # P reference is p_max[b]
                    p_ref = p_max[b] / baseMVA
                    q_ref = 0

                    # Quadratic + linear cost
                    cost_expr += a_val * (p_ref-model.Pg[b])**2 + b_val * (p_ref-model.Pg[b])

                    # Reactive power cost
                    q_val = cost_a_q[b]
                    cost_expr += q_val * (q_ref - model.Qg[b])**2

        for b in model.Buses:
            cost_expr += -barrier_alpha * pyo.log(model.V[b]-0.9)
            cost_expr += -barrier_alpha * pyo.log(1.1-model.V[b])

        return cost_expr

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model

def solve_model(model):
    """Solving the pyomo model with IPOPT solver"""
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model, tee=True) # extra info from pyomo in terminal