import numpy as np
import pandapower as pp
import pandapower.networks as pn

def extract_outputs(net):

    slack_idx = np.array(net.ext_grid["bus"])
    all_v = net.res_bus["vm_pu"]
    v = np.delete(all_v, slack_idx)

    all_theta = net.res_bus["va_degree"]
    theta = np.delete(all_theta, slack_idx)

    Ps = np.array(net.res_ext_grid["p_mw"])
    Qs = np.array(net.res_ext_grid["q_mvar"])

    I = np.array(net.res_line["i_ka"])

    y = np.concatenate([v, theta, Ps, Qs, I])
    y = y.reshape(-1, 1) # To explicitly form a column vector

    return y

if __name__ == "__main__":
    # test code
    net = pn.case9()
    pp.runpp(net)
    result = extract_outputs(net)
    print(result)
    print(len(result))

