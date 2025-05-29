import multiprocessing
from os import path
import numpy as np
import pandas as pd
import time
import torch
import math
import random
import gc
from torch.utils.tensorboard import SummaryWriter
from read_data import get_data_24, get_data_365, get_data_8760

from operate_pro import env
from operate_country import operate_365, operate_24
# 其他导入和代码...


def main():
    start = time.time()
    # 输入数据
    demand_file = r'demand.xlsx'
    wind_file = r'wind.xlsx'
    pv_file = r'pv.xlsx'
    hydro_file = r'hydro.xlsx'
    potential_file = r'potential_small.xlsx'
    line_file = r'line.xlsx'
    demand_1 = np.array(pd.read_excel(demand_file, sheet_name='2019'))
    wind = np.array(pd.read_excel(wind_file))
    pv = np.array(pd.read_excel(pv_file))
    hydro = np.array(pd.read_excel(hydro_file)) / 10
    day_max = np.array(pd.read_excel(demand_file, sheet_name='day_max'))
    # 0列是pv，1列是wind，2列是hydro,3列是nuclear
    poten = np.array(pd.read_excel(potential_file, sheet_name='poten')) * 1000
    # 0列是pv，1列是wind，4列是hydro,5列是nuclear
    existing = np.array(pd.read_excel(potential_file, sheet_name='existing'))
    # outin_range = np.array(pd.read_excel(potential_file, sheet_name='outin_range'))
    # name = np.array(pd.read_excel(potential_file, sheet_name='name'))
    line1 = np.array(pd.read_excel(line_file))
    demand_total = np.array(pd.read_excel('能研院2025-2040分省数据.xlsx', sheet_name='demand'))[:, 3] * 100000  # 2035

    demand = np.zeros((30, 8760))
    demand_rate = np.zeros(30)
    for i in range(30):
        x = np.sum(demand_1[i, :8760])
        demand_rate[i] = demand_total[i] / x
        demand[i, :] = demand_1[i, :8760] * demand_rate[i]

    distance = np.array(pd.read_excel(line_file, sheet_name='distance'))
    cost_invest_t = np.array(
        [[575, 1170, 800, 18, 41, 5], [400, 1120, 800, 18, 41, 5], [370, 1100, 800, 18, 41, 5],
         [340, 1080, 800, 18, 41, 5], [310, 1060, 800, 18, 41, 5], [280, 1040, 800, 18, 41, 5]])
    ope_cost_t = np.array([[7.5, 12.5, 105, 25], [5, 10, 135, 25], [5, 10, 150, 25], [5, 10, 165, 25],
                           [5, 10, 180, 25], [5, 10, 120, 25]]) / 1000  # 单位：thousant $/MW

    ope_cost = ope_cost_t[2, :]
    action_line_init = np.zeros((30,30))#data_24[:, 96:]


    # # 分省规划
    # Ntime = 8760
    # data_8760 = get_data_8760(Ntime, demand, wind, pv, hydro)
    # envi = env(data_8760, existing, poten, day_max, inve_cost, ope_cost)
    # c_pv_wind_control_ess_esl, cost, high_coal, C1s, h_es, carbon_all = envi.step(
    #     action=action_line_init)
    # C1s_df = pd.DataFrame(C1s)
    # capacity_df = pd.DataFrame(c_pv_wind_control_ess_esl)
    # cost_df = pd.DataFrame(cost / 1000000) # 成本单位：billion $
    # h_es_df = pd.DataFrame(h_es)
    # carbon_all_df = pd.DataFrame(carbon_all)
    # high_coal_df = pd.DataFrame(high_coal)
    # # high_coal_365_df = pd.DataFrame(high_coal_365)
    #
    # filename = 'result_withoutline_withoutes_small_nyy_2035.xlsx'
    # with pd.ExcelWriter(filename) as writer:
    #     capacity_df.to_excel(writer, sheet_name='capacity')
    #     cost_df.to_excel(writer, sheet_name='cost')
    #     h_es_df.to_excel(writer, sheet_name='duration')
    #     C1s_df.to_excel(writer, sheet_name='C1s')
    #     carbon_all_df.to_excel(writer, sheet_name='carbon')
    #     high_coal_df.to_excel(writer, sheet_name='high_coal')
    #     # high_coal_365_df.to_excel(writer, sheet_name='high_coal_365')


    # # 全国规划
    # Stime = 24
    # Ltime = 365
    # data_24 = get_data_24(Stime, demand, wind, pv, hydro, day_max, line1)
    # data_365 = get_data_365(Ltime, demand, wind, pv, hydro)
    # status, c_pv, c_wind, c_control, c_ess_s, c_esl_c, c_esl_d, c_line, cost_total, cost_pro, cost_line, p_line_365, p_line_to_24, p_line_to_365, sdm, h_es_s, h_es_l = operate_365(data_24, data_365, existing, poten, day_max, inve_cost, ope_cost, distance)
    # pro_num = 30
    # p_trans = np.zeros((pro_num, pro_num))
    # c_pv_wind_control_ess_esl = []
    # cost = []
    # h_es = []
    # print(status)
    # if status == 'optimal':
    #     x = c_pv
    #     a = np.append(x, c_wind)
    #     a = np.append(a, c_control)
    #     a = np.append(a, c_ess_s)
    #     a = np.append(a, c_esl_c)
    #     a = np.append(a, c_esl_d)
    #     c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, a)
    #     cost = np.append(cost, cost_pro)
    #     cost = np.append(cost, cost_total)
    #     cost = np.append(cost, cost_line)
    #     h_es = np.append(h_es, h_es_s)
    #     h_es = np.append(h_es, h_es_l)
    #     for i in range(pro_num):
    #         for j in range(pro_num):
    #             p_share365 = p_line_365[i, int(Ltime * j):int(Ltime * (j + 1))]
    #             p_trans[i, j] = np.sum(p_share365)
    #             if p_trans[i, j] < 0:
    #                 c_line[i, j] = -c_line[i, j]
    #     print('line: ', c_line)
    #
    #     trajectory_gen_df = pd.DataFrame(np.array(c_pv_wind_control_ess_esl).reshape(-1, pro_num).T)
    #     trajectory_carbon_df = pd.DataFrame(np.array(sdm).reshape(-1, pro_num).T)
    #     trajectory_cost_df = pd.DataFrame(np.array(cost / 1000000))  # billion $
    #     trajectory_trans_df = pd.DataFrame(np.array(c_line))
    #     trajectory_line24_df = pd.DataFrame(np.array(p_line_to_24))
    #     trajectory_line365_df = pd.DataFrame(np.array(p_line_to_365))
    #     trajectory_hes_df = pd.DataFrame(np.array(h_es).reshape(-1, pro_num).T)
    #     filename = 'result_withline_small_nyy_2035.xlsx'
    #     with pd.ExcelWriter(filename) as writer:
    #         trajectory_gen_df.to_excel(writer, sheet_name='gen')  # 单位：MW
    #         trajectory_carbon_df.to_excel(writer, sheet_name='carbon')  # 碳排单位：megatons
    #         trajectory_cost_df.to_excel(writer, sheet_name='cost')  # 成本单位：billion $
    #         trajectory_hes_df.to_excel(writer, sheet_name='duration')
    #         trajectory_line24_df.to_excel(writer, sheet_name='trans_value24')  # 单位：MW
    #         trajectory_line365_df.to_excel(writer, sheet_name='trans_value365')
    #         trajectory_trans_df.to_excel(writer, sheet_name='line')
    #


    # 灵敏度分析
    Stime = 24
    name = ['escost', 'recost', 'linecost', 'demand', 'p2']
    trajectory_gen_df = {}
    trajectory_trans_df = {}
    name_list = list(name)
    values = np.arange(0.5, 1.6, 0.1)

    for i in range(5):
        inve_cost2 = cost_invest_t[2, :].copy()
        inve_cost = inve_cost2.copy()
        p = 0.6
        demand_rate2 = 1
        for j in range(11):
            if i == 0:
                inve_cost[3:5] = inve_cost2[3:5]*values[j]
            if i == 1:
                inve_cost[:2] = inve_cost2[:2] * values[j]
            if i == 2:
                inve_cost[5] = inve_cost2[5] * values[j]
            if i == 3:
                demand_rate2 = 0.75 + 0.05*j
            if i == 4:
                p = 0.45 + 0.04*j

            data_24 = get_data_24(Stime, demand*demand_rate2, wind, pv, hydro, day_max, line1)
            # data_365 = get_data_365(Ltime, demand, wind, pv, hydro)
            status, c_pv, c_wind, c_control, c_ess_s, c_line, cost, cost_pro, cost_line, p_line_24, p_line_to_24, sdm, h_es_s = operate_24(
                data_24, existing, poten, day_max, inve_cost, ope_cost, distance, p)
            pro_num = 30
            c_pv_wind_control_ess_esl = []
            cost = []
            h_es = []
            print(status, ', cost:', inve_cost, ', demand_rate:', demand_rate2, ', p:', p)
            if status == 'optimal':
                x = c_pv
                a = np.append(x, c_wind)
                a = np.append(a, c_control)
                a = np.append(a, c_ess_s)
                c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, a)
                c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, cost_pro / 1000000)  # billion $
                # c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, cost_line / 1000000)
                c_pv_wind_control_ess_esl = np.append(c_pv_wind_control_ess_esl, h_es_s)
                trajectory_gen_df[j] = pd.DataFrame(np.array(c_pv_wind_control_ess_esl).reshape(-1, pro_num).T)

                # trajectory_carbon_df = pd.DataFrame(np.array(sdm).reshape(-1, pro_num).T)
                # trajectory_cost_df = pd.DataFrame(np.array(cost))
                trajectory_trans_df[j] = pd.DataFrame(np.array(c_line))
                # trajectory_line24_df = pd.DataFrame(np.array(p_line_to_24))
                # trajectory_hes_df = pd.DataFrame(np.array(h_es).reshape(-1, pro_num).T)
        filename = 'result_small_nyy_2035'+name_list[i]+'.xlsx'
        with pd.ExcelWriter(filename) as writer:
               for k, value in enumerate(values):
                trajectory_gen_df[k].to_excel(writer, sheet_name=f'gen{value:.1f}')  # 单位：MW
                # trajectory_carbon_df.to_excel(writer, sheet_name='carbon')  # 碳排单位：megatons
                # trajectory_cost_df.to_excel(writer, sheet_name='cost')  # 成本单位：billion $
                # trajectory_hes_df.to_excel(writer, sheet_name='duration')
                # trajectory_line24_df.to_excel(writer, sheet_name='trans_value24')  # 单位：MW
                trajectory_trans_df[k].to_excel(writer, sheet_name=f'line{value:.1f}')



    end = time.time()
    time_sum = end - start
    print(time_sum)



if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()