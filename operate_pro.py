
import numpy as np
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB, quicksum
import multiprocessing


def operate_365_GUROBI(args):
    data_24, data_365, existing, poten, day_max, inve_cost, ope_cost, line, province = args
    # Define parameters
    Stime = 24
    Ltime = 365
    p = 0.6  # 非化石能源占比	2020:0.393899204	2022:0.415420023	2025:0.464250735	2030:0.528363047
    # 2035:0.591619318	2040:0.660352711	2045:0.725916718	2050:0.8	2055:0.86979786	   2060:0.943262411

    b = 8760 / Ltime
    M = 10000000
    day_max = int(day_max[province])

    e_bat_s = 0.93
    E_bat0_s = 0
    # h_es_s = 2
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]
    a_es_s = inve_cost[3]/2
    e_bat_l_c = 0.7
    e_bat_l_d = 0.6
    E_bat0_l = 0
    # h_es_l = 20
    a_es_l = 200 * 5/20

    demand_24 = data_24[province, :24]
    wind_factor_24 = data_24[province, 24:48]
    pv_factor_24 = data_24[province, 48:72]
    hydro_factor_24 = data_24[province, 72:96]
    demand_365 = data_365[province, :Ltime]
    wind_factor_365 = data_365[province, Ltime:Ltime * 2]
    pv_factor_365 = data_365[province, Ltime * 2:Ltime * 3]
    hydro_factor_365 = data_365[province, Ltime * 3:Ltime * 4]
    line_input_total = np.sum(line, axis=0)[province]
    nuclear_cp = poten[province, 3]
    hydro_p = hydro_factor_24 * poten[province, 2]
    pv_wind_existing = existing[province, :2]
    pv_wind_potern = poten[province, :2]
    load_total_24 = np.sum(demand_24)
    load_total_365 = np.sum(demand_365)

    # Create a new model
    model = gp.Model("operate_365")
    # Suppress output
    model.Params.OutputFlag = 0

    # Create variables with lower bounds
    c_pv_wind = model.addMVar((1, 2), lb=pv_wind_existing, name="c_pv_wind")
    c_control = model.addVar(lb=0, name="c_control")
    c_ess_s = model.addVar(lb=0, name="c_ess_s")
    c_ess_l_c = model.addVar(lb=0, name="c_ess_l_c")
    c_ess_l_d = model.addVar(lb=0, name="c_ess_l_d")
    E_es_s = model.addVar(lb=0, name="E_es_s")
    E_es_l = model.addVar(lb=0, name="E_es_l")
    C1_24 = model.addMVar(Stime, lb=0, name="C1")
    higher_coal_generation24 = model.addMVar(Stime, lb=0, name="higher_coal_generation")
    # higher_wind = model.addVar(lb=0, name="higher_wind")
    p_pv_wind_control = model.addMVar((3, Stime), lb=0, name="p_pv_wind_control")
    p_charge_s = model.addMVar(Stime, lb=0, name="p_charge_s")
    p_discharge_s = model.addMVar(Stime, lb=0, name="p_discharge_s")
    E_bat_s = model.addMVar(Stime + 1, lb=0, name="E_bat_s")


    pday_pv_wind_control = model.addMVar((3, Ltime), lb=0, name="pday_pv_wind_control")
    p_charge_l = model.addMVar(Ltime, lb=0, name="p_charge_l")
    p_discharge_l = model.addMVar(Ltime, lb=0, name="p_discharge_l")
    E_bat_l = model.addMVar(Ltime + 1, lb=0, name="E_bat_l")
    higher_coal_generation365 = model.addMVar(Ltime, lb=0, name="higher_coal_generation365")
    C1_365 = model.addMVar(Ltime, lb=0, name="C1")

    p_charge_l_24 = model.addMVar(Stime, lb=0, name="p_charge_l_24")
    p_discharge_l_24 = model.addMVar(Stime, lb=0, name="p_discharge_l_24")
    E_bat_l_24 = model.addMVar(Stime + 1, lb=0, name="E_bat_l_24")

    for t in range(Stime):
        model.addConstr(
            p_pv_wind_control[0, t] + p_pv_wind_control[1, t] + p_pv_wind_control[2, t] + higher_coal_generation24[t] +
            (p_discharge_s[t] - p_charge_s[t]) +(p_discharge_l_24[t] - p_charge_l_24[t]) ==
            demand_24[t] - line_input_total - 0.7 * existing[province, 5] - hydro_factor_24[t] * existing[province, 4] + C1_24[t],
            name=f"demand_balance_24_{t}"
        )

    for d in range(Ltime):
        model.addConstr(
            pday_pv_wind_control[0, d] + pday_pv_wind_control[1, d] + pday_pv_wind_control[2, d] + higher_coal_generation365[d] +
            (p_discharge_l[d] - p_charge_l[d]) ==
            demand_365[d] - line_input_total * b - 0.7 * existing[province, 5] * b - hydro_factor_365[d] * existing[province, 4] +
            C1_365[d],
            name=f"demand_balance_365_{d}"
        )
    # model.addConstr(C1 >= 0, name="C1_non_negative")

    # Investment capacity constraints
    # model.addConstr(c_pv_wind >= pv_wind_existing, name="existing_capacity")
    model.addConstr(c_pv_wind[0, 0] <= pv_wind_potern[0], name="pv_capacity_limit")
    model.addConstr(c_pv_wind[0, 1] <= pv_wind_potern[1], name="wind_capacity_limit")
    # model.addConstr(higher_pv >= 0, name="higher_pv_non_negative")
    # model.addConstr(higher_wind >= 0, name="higher_wind_non_negative")

    # Output constraints
    for t in range(Stime):
        model.addConstr(p_pv_wind_control[0, t] <= pv_factor_24[t] * c_pv_wind[0, 0], name=f"pv_output_limit_{t}")
        model.addConstr(p_pv_wind_control[1, t] <= wind_factor_24[t] * c_pv_wind[0, 1], name=f"wind_output_limit_{t}")
        model.addConstr(p_pv_wind_control[2, t] <= c_control, name=f"control_output_limit_{t}")

    model.addConstr(
        quicksum(p_pv_wind_control[2, t] for t in range(Stime)) <= load_total_24 * (1 - p),
        name="control_output_total_limit"
    )
    for d in range(Ltime):
        model.addConstr(pday_pv_wind_control[0, d] <= pv_factor_365[d] * c_pv_wind[0, 0], name=f"pday_pv_output_limit_{d}")
        model.addConstr(pday_pv_wind_control[1, d] <= wind_factor_365[d] * c_pv_wind[0, 1], name=f"pday_wind_output_limit_{d}")
        model.addConstr(pday_pv_wind_control[2, d] <= c_control * b, name=f"pday_control_output_limit_{d}")

    model.addConstr(
        quicksum(pday_pv_wind_control[2, d] for d in range(Ltime)) <= (1 - p) * load_total_365,
        name="pday_control_output_total_limit"
    )

    # Short ES constraints
    for t in range(Stime):
        model.addConstr(p_charge_s[t] <= c_ess_s, name=f"p_charge_s_limit_{t}")
        model.addConstr(p_discharge_s[t] <= c_ess_s, name=f"p_discharge_s_limit_{t}")
        model.addConstr(p_discharge_s[t] <= E_bat_s[t], name=f"E_bat_s_discharge_limit_{t}")
        model.addConstr(E_bat_s[t + 1] == E_bat_s[t] + p_charge_s[t] * e_bat_s - p_discharge_s[t] / e_bat_s, name=f"E_bat_s_balance_{t}")

    model.addConstr(E_bat_s[0] == E_bat0_s, name="E_bat_s_initial")
    model.addConstr(E_bat_s[-1] == E_bat0_s, name="E_bat_s_final")
    for t in range(Stime + 1):
        model.addConstr(E_bat_s[t] <= E_es_s, name=f"E_bat_s_capacity_limit_{t}")

    # Long ES constraints
    model.addConstr(c_ess_l_d == c_ess_l_c, name=f"c_ess_l_d")
    for d in range(Ltime):
        model.addConstr(p_charge_l[d] <= b * c_ess_l_c, name=f"p_charge_l_limit_{d}")
        model.addConstr(p_charge_l[d] <= E_es_l, name=f"p_discharge_l_{d}")
        model.addConstr(p_discharge_l[d] <= b * c_ess_l_d, name=f"p_discharge_l_limit_{d}")
        model.addConstr(p_discharge_l[d] <= E_bat_l[d], name=f"E_bat_l_discharge_limit_{d}")
        model.addConstr(E_bat_l[d + 1] == E_bat_l[d] + p_charge_l[d] * e_bat_l_c - p_discharge_l[d] / e_bat_l_d, name=f"E_bat_l_balance_{d}")

    model.addConstr(E_bat_l[0] == E_bat0_l, name="E_bat_l_initial")
    model.addConstr(E_bat_l[-1] == E_bat0_l, name="E_bat_l_final")
    for d in range(Ltime + 1):
        model.addConstr(E_bat_l[d] <= E_es_l, name=f"E_bat_l_capacity_limit_{d}")
        model.addConstr(E_es_l >= 6 * c_ess_l_d, name=f"E_bat_l_{d}")

    # Long ES constraints on short time
    for t in range(Stime):
        model.addConstr(p_charge_l_24[t] <= c_ess_l_c, name=f"p_charge_l_24_limit_{t}")
        model.addConstr(p_discharge_l_24[t] <= c_ess_l_d, name=f"p_discharge_l_24_limit_{t}")
        model.addConstr(p_discharge_l_24[t] <= E_bat_l_24[t], name=f"E_bat_l_24_discharge_limit_{t}")
        model.addConstr(E_bat_l_24[t + 1] == E_bat_l_24[t] + p_charge_l_24[t] * e_bat_l_c - p_discharge_l_24[t] / e_bat_l_d, name=f"E_bat_l_24_balance_{t}")

    model.addConstr(E_bat_l_24[0] == E_bat_l[day_max], name="E_bat_l_24_initial")
    model.addConstr(E_bat_l_24[-1] == E_bat_l[day_max + 1], name="E_bat_l_24_final")
    for t in range(Stime + 1):
        model.addConstr(E_bat_l_24[t] <= E_es_l, name=f"E_bat_l_24_capacity_limit_{t}")

    # Define the objective function
    A2 = np.array([[a_pv], [a_wind]])
    A3 = np.ones((Stime, 1))
    A4 = np.ones((Ltime, 1))
    objective = (
            0.1168 * quicksum(c_pv_wind[0, i] * inve_cost[i] for i in range(2)) +
            0.0937 * a_control * c_control +
            18 * E_es_s + 41 * c_ess_s +
            0.0937 * (530 * (c_ess_l_c + c_ess_l_d) + 9 * E_es_l) +
            M * (quicksum(C1_24[t] + higher_coal_generation24[t] for t in range(Stime)) + quicksum(
        C1_365[d] + higher_coal_generation365[d] for d in range(Ltime))) +
            55.43 / 365 * ope_cost[2] * quicksum(pday_pv_wind_control[2, d] for d in range(Ltime)) +
            119.6 / 365 * ope_cost[0] * quicksum(pday_pv_wind_control[0, d] for d in range(Ltime)) +
            119.6 / 365 * ope_cost[1] * quicksum(pday_pv_wind_control[1, d] for d in range(Ltime))
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Optimize model
    model.optimize()

    # Collect the results
    status = model.Status
    print(status)
    # if model.status == GRB.OPTIMAL:
    c_pv_wind = c_pv_wind.X
    c_control = c_control.X
    c_ess_s = c_ess_s.X
    c_ess_l_d = c_ess_l_d.X
    c_ess_l_c = c_ess_l_c.X
    C1_24 = C1_24.X
    C1_365 = C1_365.X
    higher_coal_generation24 = higher_coal_generation24.X
    higher_coal_generation365 = higher_coal_generation365.X
    pday_pv_wind_control = pday_pv_wind_control.X
    p_coal = np.sum(pday_pv_wind_control[2, :] + higher_coal_generation365)
    # carbon1 = 365 * 0.581 * np.sum(p_pv_wind_control[2, :].X + higher_coal_generation24) / 1000000
    carbon2 = 0.581 * load_total_365 / 1000000
    SDM1 = 1 - np.sum(p_pv_wind_control[2, :].X + higher_coal_generation24)/load_total_24
    SDM2 = 1 - p_coal/load_total_365
    carbon = np.array([carbon2,SDM1,SDM2])
    cost_per = model.ObjVal-M*(C1_24 @ A3 + C1_365 @ A4 + higher_coal_generation24 @ A3 + higher_coal_generation365 @ A4)
    E_es_s = E_es_s.X
    E_es_l = E_es_l.X
    if c_ess_s>0:
        h_es_s = E_es_s*e_bat_s/c_ess_s
    else:
        h_es_s = 0
    if c_ess_l_d > 0:
        h_es_l = E_es_l*e_bat_l_d / c_ess_l_d
    else:
        h_es_l = 0

    return status, c_pv_wind, c_control, c_ess_s, c_ess_l_c, c_ess_l_d, cost_per, higher_coal_generation24, higher_coal_generation365, C1_24, C1_365, h_es_s, h_es_l, carbon


def operate_365_GUROBI_withoutes(args):
    data_24, data_365, existing, poten, day_max, inve_cost, ope_cost, line, province = args
    # Define parameters
    Stime = 24
    Ltime = 365
    p = 0.6  # 非化石能源占比	2020:0.393899204	2022:0.415420023	2025:0.464250735	2030:0.528363047
    # 2035:0.591619318	2040:0.660352711	2045:0.725916718	2050:0.8	2055:0.86979786	   2060:0.943262411

    b = 8760 / Ltime
    M = 100000
    day_max = int(day_max[province])
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]


    demand_24 = data_24[province, :24]
    wind_factor_24 = data_24[province, 24:48]
    pv_factor_24 = data_24[province, 48:72]
    hydro_factor_24 = data_24[province, 72:96]
    demand_365 = data_365[province, :Ltime]
    wind_factor_365 = data_365[province, Ltime:Ltime * 2]
    pv_factor_365 = data_365[province, Ltime * 2:Ltime * 3]
    hydro_factor_365 = data_365[province, Ltime * 3:Ltime * 4]
    line_input_total = np.sum(line, axis=0)[province]
    nuclear_cp = poten[province, 3]
    hydro_p = hydro_factor_24 * poten[province, 2]
    pv_wind_existing = existing[province, :2]
    pv_wind_potern = poten[province, :2]
    load_total_24 = np.sum(demand_24)
    load_total_365 = np.sum(demand_365)

    # Create a new model
    model = gp.Model("operate_365")
    # Suppress output
    model.Params.OutputFlag = 0

    # Create variables with lower bounds
    c_pv_wind = model.addMVar((1, 2), lb=pv_wind_existing, name="c_pv_wind")
    c_control = model.addVar(lb=0, name="c_control")
    C1_24 = model.addMVar(Stime, lb=0, name="C1")
    higher_coal_generation24 = model.addMVar(Stime, lb=0, name="higher_coal_generation")
    # higher_wind = model.addVar(lb=0, name="higher_wind")
    p_pv_wind_control = model.addMVar((3, Stime), lb=0, name="p_pv_wind_control")


    pday_pv_wind_control = model.addMVar((3, Ltime), lb=0, name="pday_pv_wind_control")
    higher_coal_generation365 = model.addMVar(Ltime, lb=0, name="higher_coal_generation365")
    C1_365 = model.addMVar(Ltime, lb=0, name="C1")


    for t in range(Stime):
        model.addConstr(
            p_pv_wind_control[0, t] + p_pv_wind_control[1, t] + p_pv_wind_control[2, t] + higher_coal_generation24[t] ==
            demand_24[t] - line_input_total - 0.7 * existing[province,5] - hydro_factor_24[t] * existing[province,4] + C1_24[t],
            name=f"demand_balance_24_{t}"
        )

    for d in range(Ltime):
        model.addConstr(
            pday_pv_wind_control[0, d] + pday_pv_wind_control[1, d] + pday_pv_wind_control[2, d] + higher_coal_generation365[d] ==
             demand_365[d] - line_input_total * b - 0.7 * existing[province,5] * b - hydro_factor_365[d] * existing[province,4] + C1_365[d],
            name=f"demand_balance_365_{d}"
        )

    # Investment capacity constraints
    model.addConstr(c_pv_wind[0, 0] <= pv_wind_potern[0], name="pv_capacity_limit")
    model.addConstr(c_pv_wind[0, 1] <= pv_wind_potern[1], name="wind_capacity_limit")

    # Output constraints
    for t in range(Stime):
        model.addConstr(p_pv_wind_control[0, t] <= pv_factor_24[t] * c_pv_wind[0, 0], name=f"pv_output_limit_{t}")
        model.addConstr(p_pv_wind_control[1, t] <= wind_factor_24[t] * c_pv_wind[0, 1], name=f"wind_output_limit_{t}")
        model.addConstr(p_pv_wind_control[2, t] <= c_control, name=f"control_output_limit_{t}")

    model.addConstr(
        quicksum(p_pv_wind_control[2, t] for t in range(Stime)) <= load_total_24 * (1 - p),
        name="control_output_total_limit"
    )
    for d in range(Ltime):
        model.addConstr(pday_pv_wind_control[0, d] <= pv_factor_365[d] * c_pv_wind[0, 0], name=f"pday_pv_output_limit_{d}")
        model.addConstr(pday_pv_wind_control[1, d] <= wind_factor_365[d] * c_pv_wind[0, 1], name=f"pday_wind_output_limit_{d}")
        model.addConstr(pday_pv_wind_control[2, d] <= c_control * b, name=f"pday_control_output_limit_{d}")

    model.addConstr(
        quicksum(pday_pv_wind_control[2, d] for d in range(Ltime)) <= (1 - p) * load_total_365,
        name="pday_control_output_total_limit"
    )


    # Define the objective function
    A3 = np.ones((Stime, 1))
    A4 = np.ones((Ltime, 1))
    objective = (
            0.1168 * quicksum(c_pv_wind[0, i] * inve_cost[i] for i in range(2)) +
            0.0937 * a_control * c_control +
            M * (quicksum(higher_coal_generation24[t] for t in range(Stime)) + quicksum(
        higher_coal_generation365[d] for d in range(Ltime))) +
            55.43 / 365 * ope_cost[2] * quicksum(pday_pv_wind_control[2, d] for d in range(Ltime)) +
            119.6 / 365 * ope_cost[0] * quicksum(pday_pv_wind_control[0, d] for d in range(Ltime)) +
            119.6 / 365 * ope_cost[1] * quicksum(pday_pv_wind_control[1, d] for d in range(Ltime))
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Optimize model
    model.optimize()

    # Collect the results
    status = model.Status
    print(status)
    # if model.status == GRB.OPTIMAL:
    c_pv_wind = c_pv_wind.X
    c_control = c_control.X
    c_ess_s = 0
    c_ess_l_d = 0
    c_ess_l_c = 0
    C1_24 = np.zeros(24)
    C1_365 = np.zeros(365)
    higher_coal_generation24 = higher_coal_generation24.X
    higher_coal_generation365 = higher_coal_generation365.X
    pday_pv_wind_control = pday_pv_wind_control.X
    p_coal = np.sum(pday_pv_wind_control[2, :] + higher_coal_generation365)
    # carbon1 = 365 * 0.581 * np.sum(p_pv_wind_control[2, :].X + higher_coal_generation24) / 1000000
    carbon2 = 0.581 * load_total_365 / 1000000
    SDM1 = 1 - np.sum(p_pv_wind_control[2, :].X + higher_coal_generation24) / load_total_24
    SDM2 = 1 - p_coal / load_total_365
    carbon = np.array([carbon2, SDM1, SDM2])
    cost_per = model.ObjVal-M*(C1_24 @ A3 + C1_365 @ A4 + higher_coal_generation24 @ A3 + higher_coal_generation365 @ A4)

    h_es_s = 0
    h_es_l = 0

    return status, c_pv_wind, c_control, c_ess_s, c_ess_l_c, c_ess_l_d, cost_per, higher_coal_generation24, higher_coal_generation365, C1_24, C1_365, h_es_s, h_es_l, carbon



class env:
    def __init__(self, data_8760, existing, poten, day_max, inve_cost, ope_cost):
        super(env, self).__init__()
        # self.data_24 = data_24
        # self.data_365 = data_365
        self.data_8760 = data_8760
        self.existing = existing
        self.poten = poten
        self.day_max = day_max
        self.inve_cost = inve_cost
        self.ope_cost = ope_cost
        # self.pro = pro

    def step(self, action):
        c_pv_wind_control_ess_esl = []
        cost_total = []
        high_coal = []
        # high_coal_365 = []
        C1s = []
        h_es = []
        carbon_all = []
        line = action
        num_provinces = 30 #30

        # 使用 multiprocessing.Pool 进行并行计算
        with multiprocessing.Pool(60) as pool:
            # 将函数参数打包成元组
            args_list = [(self.data_8760, self.existing, self.poten, self.day_max, self.inve_cost,
                          self.ope_cost, line, province) for
                         province in range(num_provinces)]

            results = pool.map(operate_8760_GUROBI_withoutes, args_list)


        for status, c_pv_wind, c_control, c_ess_s, c_ess_l_c, c_ess_l_d, cost_per, higher_coal_generation, C1, h_es_s, h_es_l, carbon in results:
            if status == 2:
                a = np.append(c_pv_wind, c_control)
                a = np.append(a, c_ess_s)
                a = np.append(a, c_ess_l_c)
                a = np.append(a, c_ess_l_d)
                c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, a)
                cost_total = np.append(cost_total, cost_per)
                carbon_all = np.append(carbon_all, carbon)
                high_coal = np.append(high_coal, higher_coal_generation)
                # high_coal_365 = np.append(high_coal_365, higher_coal_generation365)
                C1s = np.append(C1s, C1)
                # C1s = np.append(C1s, C1_365)
                h_es = np.append(h_es, h_es_s)
                h_es = np.append(h_es, h_es_l)

            else:
                print('infeasible')
                a = np.zeros((1, 6))
                c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, a)
                cost_total = np.append(cost_total, 0)
            # if np.sum(cost_total == 0) >= threshold:
            #     break

        if type(c_pv_wind_control_ess_esl).__name__ == 'list':
            c_pv_wind_control_ess_esl = np.zeros((1, 1))
        else:
            c_pv_wind_control_ess_esl = c_pv_wind_control_ess_esl.reshape((-1, 6))
            h_es = h_es.reshape((-1, 2))
            C1s = C1s.reshape((-1, 8760))
            high_coal = high_coal.reshape((-1, 8760))
            # high_coal_365 = high_coal_365.reshape((-1,365))
            carbon_all = carbon_all.reshape((-1, 3))
        return c_pv_wind_control_ess_esl, cost_total, high_coal, C1s, h_es, carbon_all




def operate_8760_GUROBI_withoutes(args):
    data_8760, existing, poten, day_max, inve_cost, ope_cost, line, province = args
    # Define parameters
    # Stime = 24
    Ltime = 8760
    p = 0.6  # 非化石能源占比	2020:0.393899204	2022:0.415420023	2025:0.464250735	2030:0.528363047
    # 2035:0.591619318	2040:0.660352711	2045:0.725916718	2050:0.8	2055:0.86979786	   2060:0.943262411

    b = 8760 / Ltime
    M = 100000
    day_max = int(day_max[province])
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]



    demand_8760 = data_8760[province, :Ltime]
    wind_factor = data_8760[province, Ltime:Ltime * 2]
    pv_factor = data_8760[province, Ltime * 2:Ltime * 3]
    hydro_factor = data_8760[province, Ltime * 3:Ltime * 4]
    line_input_total = np.sum(line, axis=0)[province]
    nuclear_cp = poten[province, 3]
    hydro_p = hydro_factor * poten[province, 2]
    pv_wind_existing = existing[province, :2]
    pv_wind_potern = poten[province, :2]
    load_total = np.sum(demand_8760)

    # Create a new model
    model = gp.Model("operate_365")
    # Suppress output
    model.Params.OutputFlag = 0

    # Create variables with lower bounds
    c_pv_wind = model.addMVar((1, 2), lb=pv_wind_existing, name="c_pv_wind")
    c_control = model.addVar(lb=0, name="c_control")
    C1 = model.addMVar(Ltime, lb=0, name="C1")
    higher_coal_generation = model.addMVar(Ltime, lb=0, name="higher_coal_generation")
    # higher_wind = model.addVar(lb=0, name="higher_wind")
    p_pv_wind_control = model.addMVar((3, Ltime), lb=0, name="p_pv_wind_control")



    for t in range(Ltime):
        model.addConstr(
            p_pv_wind_control[0, t] + p_pv_wind_control[1, t] + p_pv_wind_control[2, t] + higher_coal_generation[t] ==
            demand_8760[t] - line_input_total - 0.7 * existing[province, 5] - hydro_factor[t] * existing[province, 4] + C1[t],
            name=f"demand_balance_24_{t}"
        )

    # Investment capacity constraints
    model.addConstr(c_pv_wind[0, 0] <= pv_wind_potern[0], name="pv_capacity_limit")
    model.addConstr(c_pv_wind[0, 1] <= pv_wind_potern[1], name="wind_capacity_limit")

    # Output constraints
    for t in range(Ltime):
        model.addConstr(p_pv_wind_control[0, t] <= pv_factor[t] * c_pv_wind[0, 0], name=f"pv_output_limit_{t}")
        model.addConstr(p_pv_wind_control[1, t] <= wind_factor[t] * c_pv_wind[0, 1], name=f"wind_output_limit_{t}")
        model.addConstr(p_pv_wind_control[2, t] <= c_control, name=f"control_output_limit_{t}")

    model.addConstr(
        quicksum(p_pv_wind_control[2, t] for t in range(Ltime)) <= load_total * (1 - p),
        name="control_output_total_limit"
    )


    # Define the objective function
    A4 = np.ones((Ltime, 1))
    objective = (
            0.1168 * quicksum(c_pv_wind[0, i] * inve_cost[i] for i in range(2)) +
            0.0937 * a_control * c_control +
            M * (quicksum(higher_coal_generation[t] for t in range(Ltime))) +
            55.43 / 365 * ope_cost[2] * quicksum(p_pv_wind_control[2, t] for t in range(Ltime)) +
            119.6 / 365 * ope_cost[0] * quicksum(p_pv_wind_control[0, t] for t in range(Ltime)) +
            119.6 / 365 * ope_cost[1] * quicksum(p_pv_wind_control[1, t] for t in range(Ltime))
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Optimize model
    model.optimize()

    # Collect the results
    status = model.Status
    print(status)
    # if model.status == GRB.OPTIMAL:
    c_pv_wind = c_pv_wind.X
    c_control = c_control.X
    c_ess_s = 0
    c_ess_l_d = 0
    c_ess_l_c = 0
    C1 = C1.X
    higher_coal_generation = higher_coal_generation.X
    p_pv_wind_control = p_pv_wind_control.X
    p_re = np.sum(p_pv_wind_control[0, :] + p_pv_wind_control[1, :])
    total_re = np.sum(pv_factor * c_pv_wind[0, 0] + wind_factor * c_pv_wind[0, 1])
    re_tio = p_re / total_re
    p_coal = np.sum(p_pv_wind_control[2, :] + higher_coal_generation)
    # carbon1 = 365 * 0.581 * np.sum(p_pv_wind_control[2, :].X + higher_coal_generation) / 1000000
    carbon = 0.581 * p_coal / 1000000
    SDM = 1 - p_coal / load_total
    carbon = np.array([carbon, SDM, re_tio])
    cost_per = model.ObjVal-M*(C1 @ A4 + higher_coal_generation @ A4)

    h_es_s = 0
    h_es_l = 0

    return status, c_pv_wind, c_control, c_ess_s, c_ess_l_c, c_ess_l_d, cost_per, higher_coal_generation, C1, h_es_s, h_es_l, carbon


def operate_8760_GUROBI(args):
    data_8760, existing, poten, day_max, inve_cost, ope_cost, line, province = args
    # Define parameters
    Ntime = 8760
    p = 0.6  # 非化石能源占比	2020:0.393899204	2022:0.415420023	2025:0.464250735	2030:0.528363047
    # 2035:0.591619318	2040:0.660352711	2045:0.725916718	2050:0.8	2055:0.86979786	   2060:0.943262411

    M = 10000000
    day_max = int(day_max[province])

    e_bat_s = 0.93
    E_bat0_s = 0
    # h_es_s = 2
    a_pv = inve_cost[0]
    a_wind = inve_cost[1]
    a_control = inve_cost[2]
    a_es_s = inve_cost[3]/2
    e_bat_l_c = 0.7
    e_bat_l_d = 0.6
    E_bat0_l = 0
    # h_es_l = 20
    a_es_l = 200 * 5/20


    demand_8760 = data_8760[province, :Ntime]
    wind_factor = data_8760[province, Ntime:Ntime * 2]
    pv_factor = data_8760[province, Ntime * 2:Ntime * 3]
    hydro_factor = data_8760[province, Ntime * 3:Ntime * 4]
    line_input_total = np.sum(line, axis=0)[province]
    pv_wind_existing = existing[province, :2]
    pv_wind_potern = poten[province, :2]
    load_total = np.sum(demand_8760)

    # Create a new model
    model = gp.Model("operate_8760")
    # Suppress output
    model.Params.OutputFlag = 0

    # Create variables with lower bounds
    c_pv_wind = model.addMVar((1, 2), lb=pv_wind_existing, name="c_pv_wind")
    c_control = model.addVar(lb=0, name="c_control")
    c_ess_s = model.addVar(lb=0, name="c_ess_s")
    c_ess_l_c = model.addVar(lb=0, name="c_ess_l_c")
    c_ess_l_d = model.addVar(lb=0, name="c_ess_l_d")
    E_es_s = model.addVar(lb=0, name="E_es_s")
    E_es_l = model.addVar(lb=0, name="E_es_l")
    C1 = model.addMVar(Ntime, lb=0, name="C1")
    higher_coal_generation = model.addMVar(Ntime, lb=0, name="higher_coal_generation")
    # higher_wind = model.addVar(lb=0, name="higher_wind")
    p_pv_wind_control = model.addMVar((3, Ntime), lb=0, name="p_pv_wind_control")
    p_charge_s = model.addMVar(Ntime, lb=0, name="p_charge_s")
    p_discharge_s = model.addMVar(Ntime, lb=0, name="p_discharge_s")
    E_bat_s = model.addMVar(Ntime + 1, lb=0, name="E_bat_s")

    p_charge_l = model.addMVar(Ntime, lb=0, name="p_charge_l")
    p_discharge_l = model.addMVar(Ntime, lb=0, name="p_discharge_l")
    E_bat_l = model.addMVar(Ntime + 1, lb=0, name="E_bat_l")

    for t in range(Ntime):
        model.addConstr(
            p_pv_wind_control[0, t] + p_pv_wind_control[1, t] + p_pv_wind_control[2, t] + higher_coal_generation[t] +
            (p_discharge_s[t] - p_charge_s[t]) + (p_discharge_l[t] - p_charge_l[t]) ==
            demand_8760[t] - line_input_total - 0.7 * existing[province, 5] - hydro_factor[t] * existing[province, 4] + C1[t],
            name=f"demand_balance_{t}"
        )

    # Investment capacity constraints
    model.addConstr(c_pv_wind[0, 0] <= pv_wind_potern[0], name="pv_capacity_limit")
    model.addConstr(c_pv_wind[0, 1] <= pv_wind_potern[1], name="wind_capacity_limit")

    # Output constraints
    for t in range(Ntime):
        model.addConstr(p_pv_wind_control[0, t] <= pv_factor[t] * c_pv_wind[0, 0], name=f"pv_output_limit_{t}")
        model.addConstr(p_pv_wind_control[1, t] <= wind_factor[t] * c_pv_wind[0, 1], name=f"wind_output_limit_{t}")
        model.addConstr(p_pv_wind_control[2, t] <= c_control, name=f"control_output_limit_{t}")

    model.addConstr(
        quicksum(p_pv_wind_control[2, t] for t in range(Ntime)) <= load_total * (1 - p),
        name="control_output_total_limit"
    )


    # Short ES constraints
    for t in range(Ntime):
        model.addConstr(p_charge_s[t] <= c_ess_s, name=f"p_charge_s_limit_{t}")
        model.addConstr(p_discharge_s[t] <= c_ess_s, name=f"p_discharge_s_limit_{t}")
        model.addConstr(p_discharge_s[t] <= E_bat_s[t], name=f"E_bat_s_discharge_limit_{t}")
        model.addConstr(E_bat_s[t + 1] == E_bat_s[t] + p_charge_s[t] * e_bat_s - p_discharge_s[t] / e_bat_s, name=f"E_bat_s_balance_{t}")

    model.addConstr(E_bat_s[0] == E_bat0_s, name="E_bat_s_initial")
    model.addConstr(E_bat_s[-1] == E_bat0_s, name="E_bat_s_final")
    for t in range(Ntime + 1):
        model.addConstr(E_bat_s[t] <= E_es_s, name=f"E_bat_s_capacity_limit_{t}")

    # Long ES constraints on short time
    model.addConstr(c_ess_l_d == c_ess_l_c, name=f"c_ess_l_d")
    for t in range(Ntime):
        model.addConstr(p_charge_l[t] <= c_ess_l_c, name=f"p_charge_l_24_limit_{t}")
        model.addConstr(p_discharge_l[t] <= c_ess_l_d, name=f"p_discharge_l_24_limit_{t}")
        model.addConstr(p_discharge_l[t] <= E_bat_l[t], name=f"E_bat_l_24_discharge_limit_{t}")
        model.addConstr(E_bat_l[t + 1] == E_bat_l[t] + p_charge_l[t] * e_bat_l_c - p_discharge_l[t] / e_bat_l_d, name=f"E_bat_l_balance_{t}")

    model.addConstr(E_bat_l[0] == E_bat0_l, name="E_bat_l_initial")
    model.addConstr(E_bat_l[-1] == E_bat0_l, name="E_bat_l_final")
    for t in range(Ntime + 1):
        model.addConstr(E_bat_l[t] <= E_es_l, name=f"E_bat_l_capacity_limit_{t}")
    model.addConstr(E_es_l >= 6 * c_ess_l_d, name=f"E_bat_l")

    # Define the objective function
    A2 = np.array([[a_pv], [a_wind]])
    A3 = np.ones((Ntime, 1))
    objective = (
            0.1168 * quicksum(c_pv_wind[0, i] * inve_cost[i] for i in range(2)) +
            0.0937 * a_control * c_control +
            18 * E_es_s + 41 * c_ess_s +
            0.0937 * (530 * (c_ess_l_c + c_ess_l_d) + 9 * E_es_l) +
            M * (quicksum(C1[t] + higher_coal_generation[t] for t in range(Ntime))) +
            55.43 / 365 * ope_cost[2] * quicksum(p_pv_wind_control[2, d] for d in range(Ntime)) +
            119.6 / 365 * ope_cost[0] * quicksum(p_pv_wind_control[0, d] for d in range(Ntime)) +
            119.6 / 365 * ope_cost[1] * quicksum(p_pv_wind_control[1, d] for d in range(Ntime))
    )
    model.setObjective(objective, GRB.MINIMIZE)

    # Optimize model
    model.optimize()

    # Collect the results
    status = model.Status
    print(status)
    # if model.status == GRB.OPTIMAL:
    c_pv_wind = c_pv_wind.X
    c_control = c_control.X
    c_ess_s = c_ess_s.X
    c_ess_l_d = c_ess_l_d.X
    c_ess_l_c = c_ess_l_c.X
    C1 = C1.X
    higher_coal_generation = higher_coal_generation.X
    p_pv_wind_control = p_pv_wind_control.X
    p_re = np.sum(p_pv_wind_control[0, :] +p_pv_wind_control[1, :])
    total_re = np.sum(pv_factor * c_pv_wind[0, 0] + wind_factor * c_pv_wind[0, 1])
    re_tio = p_re/total_re
    p_coal = np.sum(p_pv_wind_control[2, :] + higher_coal_generation)
    # carbon1 = 365 * 0.581 * np.sum(p_pv_wind_control[2, :].X + higher_coal_generation) / 1000000
    carbon = 0.581 * p_coal / 1000000
    SDM = 1 - p_coal / load_total
    carbon = np.array([carbon,SDM,re_tio])
    cost_per = model.ObjVal-M*(C1 @ A3 + higher_coal_generation @ A3)
    E_es_s = E_es_s.X
    E_es_l = E_es_l.X
    if c_ess_s>0:
        h_es_s = E_es_s*e_bat_s/c_ess_s
    else:
        h_es_s = 0
    if c_ess_l_d > 0:
        h_es_l = E_es_l*e_bat_l_d / c_ess_l_d
    else:
        h_es_l = 0

    return status, c_pv_wind, c_control, c_ess_s, c_ess_l_c, c_ess_l_d, cost_per, higher_coal_generation, C1, h_es_s, h_es_l, carbon

