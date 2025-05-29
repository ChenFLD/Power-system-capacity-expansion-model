import numpy as np
import pandas as pd
import cvxpy as cp


def operate_365(data_24, data_365, existing, poten, day_max, inve_cost, ope_cost, dis_pro):
    Stime=24
    Ltime=365
    p = 0.6  # 可再生能源占比
    b = np.array(8760 / Ltime, dtype='int')
    M = 1000000000000
    day_max = np.array(day_max, dtype='int')
    pro_num = 30

    e_bat_s = 0.93
    E_bat0_s = 0
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]
    e_bat_l_c = 0.7
    e_bat_l_d = 0.6
    E_bat0_l = 0

    demand_24 = data_24[:, 0:24]
    wind_factor_24 = data_24[:, 24:48]
    pv_factor_24 = data_24[:, 48:72]
    hydro_factor_24 = data_24[:, 72:96]
    demand_365 = data_365[:, 0:Ltime]
    wind_factor_365 = data_365[:, Ltime:Ltime * 2]
    pv_factor_365 = data_365[:, Ltime * 2:Ltime * 3]
    hydro_factor_365 = data_365[:, Ltime * 3:Ltime * 4]
    load_total_24 = np.sum(demand_24)
    load_total_24pro = np.sum(demand_24, axis=1)
    load_total_365 = np.sum(demand_365)
    load_total_365pro = np.sum(demand_365, axis=1)

    # creat variable
    c_pv = cp.Variable(pro_num)
    c_wind = cp.Variable(pro_num)
    c_control = cp.Variable(pro_num)
    c_ess_s = cp.Variable(pro_num)
    E_ess = cp.Variable(pro_num)
    c_esl_c = cp.Variable(pro_num)
    c_esl_d = cp.Variable(pro_num)
    E_esl = cp.Variable(pro_num)
    c_line = cp.Variable((pro_num,pro_num))
    p_pv = cp.Variable((pro_num, Stime))
    p_wind = cp.Variable((pro_num, Stime))
    # p_hydro = cp.Variable((pro_num, Stime))
    # p_nuclear = cp.Variable((pro_num, Stime))
    p_control = cp.Variable((pro_num, Stime))
    p_charge_s = cp.Variable((pro_num, Stime))
    p_discharge_s = cp.Variable((pro_num, Stime))
    p_bat_s = cp.Variable((pro_num, Stime))
    E_bat_s = cp.Variable((pro_num, Stime + 1))


    # 模拟长期储能
    pday_pv = cp.Variable((pro_num, Ltime))
    pday_wind = cp.Variable((pro_num, Ltime))
    pday_control = cp.Variable((pro_num, Ltime))
    # pday_hydro = cp.Variable((pro_num, Ltime))
    # pday_nuclear = cp.Variable((pro_num, Ltime))
    p_charge_l = cp.Variable((pro_num, Ltime))
    p_discharge_l = cp.Variable((pro_num, Ltime))
    p_bat_l = cp.Variable((pro_num, Ltime))
    E_bat_l = cp.Variable((pro_num, Ltime + 1))

    p_charge_l_24 = cp.Variable((pro_num, Stime))
    p_discharge_l_24 = cp.Variable((pro_num, Stime))
    p_bat_l_24 = cp.Variable((pro_num, Stime))
    E_bat_l_24 = cp.Variable((pro_num, Stime + 1))

    p_line_24 = cp.Variable((pro_num, pro_num * Stime))
    p_line_to_24 = cp.Variable((pro_num, Stime))
    p_line_365 = cp.Variable((pro_num, pro_num * Ltime))
    p_line_to_365 = cp.Variable((pro_num, Ltime))

    # create constraints
    constraints = []
    # demand balance
    b1 = np.ones((1, pro_num))
    for province in range(pro_num):
        for t in range(Stime):
            constraints = np.append(constraints,
                                    p_line_to_24[province, t] == b1 @ p_line_24[:, int(Stime * province + t)])
            constraints = np.append(constraints,
                                    p_pv[province, t] + p_wind[province, t] + p_control[province, t] + p_bat_s[
                                        province, t] + p_bat_l_24[province, t] + p_line_to_24[province, t] ==
                                    demand_24[province, t] - hydro_factor_24[province, t] * existing[province, 4]
                                    - 0.7 * existing[province, 5])
        for d in range(Ltime):
            constraints = np.append(constraints,
                                    p_line_to_365[province, d] == b1 @ p_line_365[:, int(Ltime * province + d)])
            constraints = np.append(constraints,
                                    pday_pv[province, d] + pday_wind[province, d] + pday_control[province, d] + p_bat_l[
                                        province, d] + p_line_to_365[province, d] == demand_365[province, d]
                                    -hydro_factor_365[province, d] * existing[province, 4] - 0.7* existing[province, 5] * b)

    # Investment capacity constraints
    constraints = np.append(constraints, c_pv >= existing[:, 0])
    constraints = np.append(constraints, c_pv <= poten[:, 0])
    constraints = np.append(constraints, c_wind >= existing[:, 1])
    constraints = np.append(constraints, c_wind <= poten[:, 1])
    constraints = np.append(constraints, c_line >= 0)


    # transmission line constraints
    for i in range(pro_num):
        for j in range(pro_num):
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] == -p_line_24[j,
                                                                                                        int(Stime * i): int(Stime * (i + 1))])
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] >= -c_line[i,j])
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] <= c_line[i,j])

            constraints = np.append(constraints, p_line_365[i, int(Ltime * j): int(Ltime * (j + 1))] == -p_line_365[j,
                                                                                                         int(Ltime * i): int(Ltime * (i + 1))])
            constraints = np.append(constraints, p_line_365[i, int(Ltime * j): int(Ltime * (j + 1))] >= -c_line[i,j] * b)
            constraints = np.append(constraints, p_line_365[i, int(Ltime * j): int(Ltime * (j + 1))] <= c_line[i,j] * b)

    # output constraints
    for t in range(Stime):
        for province in range(pro_num):
            constraints = np.append(constraints, p_pv[province, t] >= 0)
            constraints = np.append(constraints, p_pv[province, t] <= pv_factor_24[province, t] * c_pv[province])
            constraints = np.append(constraints, p_wind[province, t] <= wind_factor_24[province, t] * c_wind[province])
            constraints = np.append(constraints, p_wind[province, t] >= 0)

            constraints = np.append(constraints, p_control[province, t] <= c_control[province])
            constraints = np.append(constraints, p_control[province, t] >= 0)
    constraints = np.append(constraints, cp.sum(p_control, axis=1) <= load_total_24 * (1 - p))
    for province in range(pro_num):
        constraints = np.append(constraints,
                                p_control[province, :] @ np.ones((Stime, 1)) <= (1 - p) * load_total_24pro[
                                    province])

    for d in range(Ltime):
        for province in range(pro_num):
            constraints = np.append(constraints, pday_pv[province, d] >= 0)
            constraints = np.append(constraints, pday_pv[province, d] <= pv_factor_365[province, d] * c_pv[province])
            constraints = np.append(constraints,
                                    pday_wind[province, d] <= wind_factor_365[province, d] * c_wind[province])
            constraints = np.append(constraints, pday_wind[province, d] >= 0)

            constraints = np.append(constraints, pday_control[province, d] <= c_control[province] * b)
            constraints = np.append(constraints, pday_control[province, d] >= 0)
    constraints = np.append(constraints,
                            cp.sum(pday_control, axis=1) <= (1 - p) * load_total_365)
    for province in range(pro_num):
        constraints = np.append(constraints,
                                pday_control[province, :] @ np.ones((Ltime, 1)) <= (1 - p) * load_total_365pro[
                                    province])

    # short ES constraints
    constraints = np.append(constraints, p_charge_s >= np.zeros((pro_num, Stime)))
    constraints = np.append(constraints, p_discharge_s >= np.zeros((pro_num, Stime)))
    # constraints = np.append(constraints, Index_cha_s + Index_dis_s == np.ones((1, Stime)))
    for province in range(pro_num):
        constraints = np.append(constraints, p_charge_s[province, :] <= c_ess_s[province])
        constraints = np.append(constraints, p_discharge_s[province, :] <= c_ess_s[province])
        for t in range(Stime):
            constraints = np.append(constraints, E_bat_s[province, t + 1] == E_bat_s[province, t] + p_charge_s[
                province, t] * e_bat_s - p_discharge_s[province, t] / e_bat_s)
            constraints = np.append(constraints, p_discharge_s[:, t] <= E_bat_s[:, t])
    constraints = np.append(constraints, p_bat_s == p_discharge_s - p_charge_s)
    constraints = np.append(constraints, E_bat_s >= np.zeros((pro_num, Stime + 1)))
    for province in range(pro_num):
        constraints = np.append(constraints, E_bat_s[province, 0] == E_bat0_s)
        constraints = np.append(constraints, E_bat_s[province, -1] == E_bat0_s)
        for t in range(Stime + 1):
            constraints = np.append(constraints, E_bat_s[province, t] <= E_ess[province])

    # long ES constraints
    constraints = np.append(constraints, c_esl_c == c_esl_d)
    constraints = np.append(constraints, p_charge_l >= np.zeros((pro_num, Ltime)))
    constraints = np.append(constraints, p_discharge_l >= np.zeros((pro_num, Ltime)))
    # constraints = np.append(constraints, Index_cha_s + Index_dis_s == np.ones((1, Stime)))
    for province in range(pro_num):
        constraints = np.append(constraints, p_charge_l[province, :] <= c_esl_c[province] * b)
        constraints = np.append(constraints, p_charge_l[province, :] <= E_esl[province])
        constraints = np.append(constraints, p_discharge_l[province, :] <= c_esl_d[province] * b)

        for d in range(Ltime):
            constraints = np.append(constraints, E_bat_l[province, d + 1] == E_bat_l[province, d] + p_charge_l[
                province, d] * e_bat_l_c - p_discharge_l[province, d] / e_bat_l_d)
            constraints = np.append(constraints, p_discharge_l[:, d] <= E_bat_l[:, d])
    constraints = np.append(constraints, p_bat_l == p_discharge_l - p_charge_l)
    constraints = np.append(constraints, E_bat_l >= np.zeros((pro_num, Ltime + 1)))
    for province in range(pro_num):
        constraints = np.append(constraints, E_bat_l[province, 0] == E_bat0_l)
        constraints = np.append(constraints, E_bat_l[province, -1] == E_bat0_l)
        for d in range(Ltime + 1):
            constraints = np.append(constraints, E_bat_l[province, d] <= E_esl[province])

    # long ES constraints on short time
    constraints = np.append(constraints, p_charge_l_24 >= np.zeros((pro_num, Stime)))
    constraints = np.append(constraints, p_discharge_l_24 >= np.zeros((pro_num, Stime)))
    # constraints = np.append(constraints, Index_cha_s + Index_dis_s == np.ones((1, Stime)))
    for province in range(pro_num):
        constraints = np.append(constraints, p_charge_l_24[province, :] <= c_esl_c[province])
        constraints = np.append(constraints, p_discharge_l_24[province, :] <= c_esl_c[province])

        for t in range(Stime):
            constraints = np.append(constraints, E_bat_l_24[province, t + 1] == E_bat_l_24[province, t] + p_charge_s[
                province, t] * e_bat_l_c - p_discharge_l_24[province, t] / e_bat_l_d)
            constraints = np.append(constraints, p_discharge_l_24[:, t] <= E_bat_l_24[:, t])
    constraints = np.append(constraints, p_bat_l_24 == p_discharge_l_24 - p_charge_l_24)
    constraints = np.append(constraints, E_bat_l_24 >= np.zeros((pro_num, Stime + 1)))
    for province in range(pro_num):
        constraints = np.append(constraints, E_bat_l_24[province, 0] == E_bat_l[0, day_max[province]])
        constraints = np.append(constraints, E_bat_l_24[province, -1] == E_bat_l[0, day_max[province] + 1])
        for t in range(Stime + 1):
            constraints = np.append(constraints, E_bat_l_24[province, t] <= E_esl[province])

    # objective function
    cost_per_province = []
    for i in range(pro_num):
        # Calculate cost for each province i
        cost_pv = 0.1168 * a_pv * c_pv[i]  # PV cost for province i
        cost_wind = 0.1168 * a_wind * c_wind[i]  # Wind cost for province i
        cost_control = 0.0937 * a_control * c_control[i]  # Control system cost for province i
        cost_es_s = 18 * E_ess[i] + 41* c_ess_s[i]  # Short-term storage cost for province i
        cost_es_l = 0.0937 * (530 * (c_esl_c[i] + c_esl_d[i]) + 9 * E_esl[i])  # Long-term storage cost for province i
        cost_op_control = 55.43 / 365 * ope_cost[2] * cp.sum(
            pday_control[i, :])  # Operational control cost for province i
        cost_op_pv = 119.6 / 365 * ope_cost[0] * cp.sum(pday_pv[i, :])  # Operational PV cost for province i
        cost_op_wind = 119.6 / 365 * ope_cost[1] * cp.sum(pday_wind[i, :])  # Operational Wind cost for province i

        # Summing up the costs for the current province i
        cost_per_i = (
                cost_pv + cost_wind + cost_control + cost_es_s + cost_es_l +
                cost_op_control + cost_op_pv + cost_op_wind
        )

        # Append the cost for province i to the list
        cost_per_province.append(cost_per_i)
    # Convert the list into a CVXPY object if needed
    cost_per_province = cp.vstack(cost_per_province)
    cost_line = 0.0937 * 5/6.9 * cp.sum(cp.multiply(c_line, dis_pro)) / 2

    objective = cp.Minimize(cp.sum(cost_per_province) + cost_line)

    # thousand $


    prob = cp.Problem(objective, constraints)
    prob.solve(solver='CPLEX')
    # prob.solve(solver=cp.CPLEX, verbose=True, cplex_filename='model.lp', reoptimize=True)
    pday_pv = pday_pv.value
    pday_wind = pday_wind.value
    pday_control = pday_control.value
    p_bat_l = p_bat_l.value
    # pday_hydro = pday_hydro.value
    # pday_nuclear = pday_nuclear.value
    p_line_365 = p_line_365.value
    # x=np.sum(pday_pv, axis=1)+np.sum(pday_wind, axis=1)+np.sum(pday_control, axis=1)+np.sum(pday_nuclear, axis=1)+np.sum(pday_hydro, axis=1)
    print(prob.status)
    # p_curve = np.vstack((pday_pv, pday_wind, pday_control, p_bat_l, pday_hydro, pday_nuclear))
    c_pv = c_pv.value
    c_wind = c_wind.value
    c_control = c_control.value
    c_ess_s = c_ess_s.value
    c_esl_c = c_esl_c.value
    c_esl_d = c_esl_d.value
    c_line = c_line.value
    p_charge_l = p_charge_s.value
    p_discharge_l = p_discharge_s.value
    p_line_to_24 = p_line_to_24.value
    p_line_to_365 = p_line_to_365.value
    # net_load = demand_24
    cost_pro = cost_per_province.value
    cost_line = cost_line.value
    status = prob.status
    cost = prob.value
    E_ess = E_ess.value
    E_esl = E_esl.value
    h_es_s = np.zeros(pro_num)
    h_es_l = np.zeros(pro_num)
    for i in range(pro_num):
        if c_ess_s[i] > 0:
            h_es_s[i] = E_ess[i]*e_bat_s/c_ess_s[i]

        if c_esl_d[i] > 0:
            h_es_l[i] = E_esl[i]*e_bat_l_d / c_esl_d[i]
    re_tio = np.zeros(30)
    for pro in range(30):
        p_re = np.sum(pday_pv[pro,:] + pday_wind[pro,:])
        total_re = np.sum(pv_factor_365[pro,:] * c_pv[pro] + wind_factor_365[pro,:] * c_wind[pro])
        re_tio[pro] = p_re / total_re
    carbon_pro = 0.581 * load_total_365pro / 1000000
    sdm_365 = 1 - np.sum(pday_control, axis=1) / load_total_365pro
    sdm_24 = 1 - np.sum(p_control.value, axis=1) / load_total_24pro
    sdm = []
    sdm = np.append(sdm, carbon_pro)
    sdm = np.append(sdm, sdm_24)
    sdm = np.append(sdm, sdm_365)
    sdm = np.append(sdm, re_tio)

    return status, c_pv, c_wind, c_control, c_ess_s, c_esl_c, c_esl_d, c_line, cost, cost_pro, cost_line, p_line_365, p_line_to_24, p_line_to_365, sdm, h_es_s, h_es_l



def operate_24(data_24, existing, poten, day_max, inve_cost, ope_cost, dis_pro, p):
    Stime=24
    Ltime=365
    # p = 0.6  # 可再生能源占比
    b = np.array(8760 / Ltime, dtype='int')
    # M = 1000000000000
    day_max = np.array(day_max, dtype='int')
    pro_num = 30

    e_bat_s = 0.93
    E_bat0_s = 0
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]
    a_E_ess = inve_cost[3]
    a_c_ess = inve_cost[4]
    a_line = inve_cost[5]

    demand_24 = data_24[:, 0:24]
    wind_factor_24 = data_24[:, 24:48]
    pv_factor_24 = data_24[:, 48:72]
    hydro_factor_24 = data_24[:, 72:96]
    load_total_24 = np.sum(demand_24)
    load_total_24pro = np.sum(demand_24, axis=1)

    # creat variable
    c_pv = cp.Variable(pro_num)
    c_wind = cp.Variable(pro_num)
    c_control = cp.Variable(pro_num)
    c_ess_s = cp.Variable(pro_num)
    E_ess = cp.Variable(pro_num)
    # c_esl_c = cp.Variable(pro_num)
    # c_esl_d = cp.Variable(pro_num)
    # E_esl = cp.Variable(pro_num)
    c_line = cp.Variable((pro_num,pro_num))
    p_pv = cp.Variable((pro_num, Stime))
    p_wind = cp.Variable((pro_num, Stime))
    # p_hydro = cp.Variable((pro_num, Stime))
    # p_nuclear = cp.Variable((pro_num, Stime))
    p_control = cp.Variable((pro_num, Stime))
    p_charge_s = cp.Variable((pro_num, Stime))
    p_discharge_s = cp.Variable((pro_num, Stime))
    p_bat_s = cp.Variable((pro_num, Stime))
    E_bat_s = cp.Variable((pro_num, Stime + 1))


    p_line_24 = cp.Variable((pro_num, pro_num * Stime))
    p_line_to_24 = cp.Variable((pro_num, Stime))
    # p_line_365 = cp.Variable((pro_num, pro_num * Ltime))
    # p_line_to_365 = cp.Variable((pro_num, Ltime))

    # create constraints
    constraints = []
    # demand balance
    b1 = np.ones((1, pro_num))
    for province in range(pro_num):
        for t in range(Stime):
            constraints = np.append(constraints,
                                    p_line_to_24[province, t] == b1 @ p_line_24[:, int(Stime * province + t)])
            constraints = np.append(constraints,
                                    p_pv[province, t] + p_wind[province, t] + p_control[province, t] + p_bat_s[
                                        province, t] + p_line_to_24[province, t] ==
                                    demand_24[province, t] - hydro_factor_24[province, t] * existing[province, 4]
                                    - 0.7 * existing[province, 5])


    # Investment capacity constraints
    constraints = np.append(constraints, c_pv >= existing[:, 0])
    constraints = np.append(constraints, c_pv <= poten[:, 0])
    constraints = np.append(constraints, c_wind >= existing[:, 1])
    constraints = np.append(constraints, c_wind <= poten[:, 1])
    constraints = np.append(constraints, c_line >= 0)


    # transmission line constraints
    for i in range(pro_num):
        for j in range(pro_num):
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] == -p_line_24[j,
                                                                                                        int(Stime * i): int(Stime * (i + 1))])
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] >= -c_line[i,j])
            constraints = np.append(constraints, p_line_24[i, int(Stime * j): int(Stime * (j + 1))] <= c_line[i,j])


    # output constraints
    for t in range(Stime):
        for province in range(pro_num):
            constraints = np.append(constraints, p_pv[province, t] >= 0)
            constraints = np.append(constraints, p_pv[province, t] <= pv_factor_24[province, t] * c_pv[province])
            constraints = np.append(constraints, p_wind[province, t] <= wind_factor_24[province, t] * c_wind[province])
            constraints = np.append(constraints, p_wind[province, t] >= 0)

            constraints = np.append(constraints, p_control[province, t] <= c_control[province])
            constraints = np.append(constraints, p_control[province, t] >= 0)
    constraints = np.append(constraints, cp.sum(p_control, axis=1) <= load_total_24 * (1 - p))
    for province in range(pro_num):
        constraints = np.append(constraints,
                                p_control[province, :] @ np.ones((Stime, 1)) <= (1 - p) * load_total_24pro[
                                    province])


    # short ES constraints
    constraints = np.append(constraints, p_charge_s >= np.zeros((pro_num, Stime)))
    constraints = np.append(constraints, p_discharge_s >= np.zeros((pro_num, Stime)))
    # constraints = np.append(constraints, Index_cha_s + Index_dis_s == np.ones((1, Stime)))
    for province in range(pro_num):
        constraints = np.append(constraints, p_charge_s[province, :] <= c_ess_s[province])
        constraints = np.append(constraints, p_discharge_s[province, :] <= c_ess_s[province])
        for t in range(Stime):
            constraints = np.append(constraints, E_bat_s[province, t + 1] == E_bat_s[province, t] + p_charge_s[
                province, t] * e_bat_s - p_discharge_s[province, t] / e_bat_s)
            constraints = np.append(constraints, p_discharge_s[:, t] <= E_bat_s[:, t])
    constraints = np.append(constraints, p_bat_s == p_discharge_s - p_charge_s)
    constraints = np.append(constraints, E_bat_s >= np.zeros((pro_num, Stime + 1)))
    for province in range(pro_num):
        constraints = np.append(constraints, E_bat_s[province, 0] == E_bat0_s)
        constraints = np.append(constraints, E_bat_s[province, -1] == E_bat0_s)
        for t in range(Stime + 1):
            constraints = np.append(constraints, E_bat_s[province, t] <= E_ess[province])


    # objective function
    cost_per_province = []
    for i in range(pro_num):
        # Calculate cost for each province i
        cost_pv = 0.1168 * a_pv * c_pv[i]  # PV cost for province i
        cost_wind = 0.1168 * a_wind * c_wind[i]  # Wind cost for province i
        cost_control = 0.0937 * a_control * c_control[i]  # Control system cost for province i
        cost_es_s = a_E_ess * E_ess[i] + a_c_ess* c_ess_s[i]  # Short-term storage cost for province i
        # cost_es_l = 0.0937 * (530 * (c_esl_c[i] + c_esl_d[i]) + 9 * E_esl[i])  # Long-term storage cost for province i
        cost_op_control = 55.43 * ope_cost[2] * cp.sum(p_control[i, :])  # Operational control cost for province i
        cost_op_pv = 119.6 * ope_cost[0] * cp.sum(p_pv[i, :])  # Operational PV cost for province i
        cost_op_wind = 119.6 * ope_cost[1] * cp.sum(p_wind[i, :])  # Operational Wind cost for province i

        # Summing up the costs for the current province i
        cost_per_i = (
                cost_pv + cost_wind + cost_control + cost_es_s + # cost_es_l +
                cost_op_control + cost_op_pv + cost_op_wind
        )

        # Append the cost for province i to the list
        cost_per_province.append(cost_per_i)
    # Convert the list into a CVXPY object if needed
    cost_per_province = cp.vstack(cost_per_province)
    cost_line = 0.0937 * a_line/6.9 * cp.sum(cp.multiply(c_line, dis_pro)) / 2

    objective = cp.Minimize(cp.sum(cost_per_province) + cost_line)

    # thousand $


    prob = cp.Problem(objective, constraints)
    prob.solve(solver='CPLEX')
    # prob.solve(solver=cp.CPLEX, verbose=True, cplex_filename='model.lp', reoptimize=True)
    p_pv = p_pv.value
    p_wind = p_wind.value
    p_control = p_control.value

    # x=np.sum(pday_pv, axis=1)+np.sum(pday_wind, axis=1)+np.sum(pday_control, axis=1)+np.sum(pday_nuclear, axis=1)+np.sum(pday_hydro, axis=1)
    # print(prob.status)
    # p_curve = np.vstack((pday_pv, pday_wind, pday_control, p_bat_l, pday_hydro, pday_nuclear))
    c_pv = c_pv.value
    c_wind = c_wind.value
    c_control = c_control.value
    c_ess_s = c_ess_s.value
    c_line = c_line.value
    p_line_24 = p_line_24.value
    p_line_to_24 = p_line_to_24.value
    # net_load = demand_24
    cost_pro = cost_per_province.value
    cost_line = cost_line.value
    status = prob.status
    cost = prob.value
    E_ess = E_ess.value
    # E_esl = E_esl.value
    h_es_s = np.zeros(pro_num)
    for i in range(pro_num):
        if c_ess_s[i] > 0:
            h_es_s[i] = E_ess[i]*e_bat_s/c_ess_s[i]

        # if c_esl_d[i] > 0:
        #     h_es_l[i] = E_esl[i]*e_bat_l_d / c_esl_d[i]
    re_tio = np.zeros(30)
    for pro in range(30):
        p_re = np.sum(p_pv[pro,:] + p_wind[pro,:])
        total_re = np.sum(pv_factor_24[pro,:] * c_pv[pro] + wind_factor_24[pro,:] * c_wind[pro])
        re_tio[pro] = p_re / total_re
    # carbon_pro = 0.581 * load_total_24pro / 1000000
    # sdm_365 = 1 - np.sum(pday_control, axis=1) / load_total_365pro
    sdm_24 = 1 - np.sum(p_control, axis=1) / load_total_24pro
    sdm = []
    # sdm = np.append(sdm, carbon_pro)
    sdm = np.append(sdm, sdm_24)
    # sdm = np.append(sdm, sdm_365)
    sdm = np.append(sdm, re_tio)

    return status, c_pv, c_wind, c_control, c_ess_s, c_line, cost, cost_pro, cost_line, p_line_24, p_line_to_24, sdm, h_es_s

