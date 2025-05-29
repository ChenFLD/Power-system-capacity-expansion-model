import numpy as np
import pandas as pd
import os.path as path



def get_data_8760(Ntime, demand_1, wind, pv, hydro):
    demand = demand_1[:, :Ntime]
    wind_factor = wind[:, :Ntime]
    pv_factor = pv[:, :Ntime]
    hydro_factor = hydro[:Ntime]
    demand = np.array(list(demand)).reshape(-1,Ntime)
    wind_factor = np.array(list(wind_factor)).reshape(-1,Ntime)
    pv_factor = np.array(list(pv_factor)).reshape(-1,Ntime)
    hydro_factor = np.array(list(hydro_factor)).reshape(-1,Ntime)
    hydro_factor = np.tile(hydro_factor, (30, 1))
    # line_init = np.nan_to_num(line1)
    data_8760 = np.concatenate((demand, wind_factor, pv_factor), axis=1)
    data_8760 = np.concatenate((data_8760, hydro_factor), axis=1)

    return data_8760

def get_data_24(Ntime, demand_1, wind, pv, hydro, day_max, line1):
    demand=[]
    wind_factor=[]
    pv_factor =[]
    hydro_factor =[]
    # pro = np.array([4,14,21], dtype='int')  # 内蒙，山东，重庆

    for x in range(30):
        index_max=np.arange(day_max[x]*Ntime,(day_max[x]+1)*Ntime,1)
        index_max.dtype='int64'
        demand= np.append(demand, demand_1[x*np.ones(Ntime,dtype='int'), index_max])  # 2020 1.040999118	2025 1.281692556  2030 1.420091283	 2035 1.504333986	2040 1.540438002	2045 1.552472674	2050 1.55849001

        wind_factor = np.append(wind_factor, wind[x*np.ones(Ntime,dtype='int'), index_max])
        pv_factor = np.append(pv_factor, pv[x*np.ones(Ntime,dtype='int'), index_max])
        hydro_factor = np.append(hydro_factor, hydro[index_max])
    demand = np.array(list(demand)).reshape(-1,Ntime)
    wind_factor = np.array(list(wind_factor)).reshape(-1,Ntime)
    pv_factor = np.array(list(pv_factor)).reshape(-1,Ntime)
    hydro_factor = np.array(list(hydro_factor)).reshape(-1,Ntime)
    line_init=np.nan_to_num(line1)
    data_24 = np.concatenate((demand, wind_factor, pv_factor), axis=1)
    data_24 = np.concatenate((data_24, hydro_factor, line_init), axis=1)

    return data_24

def get_data_365(Ltime, demand_1, wind, pv, hydro):
    demand_365 = []
    wind_365 = []
    pv_365 = []
    hydro_365 = []
    b=np.array(8760/Ltime, dtype='int')
    # pro = np.array([4,14,21], dtype='int') # 内蒙，山东，重庆
    pro_num = 30

    for i in range(pro_num):
        for d in range(Ltime):
            demand_365 = np.append(demand_365, np.sum(demand_1[i, d * b:(d + 1) * b]))
            wind_365 = np.append(wind_365, np.sum(wind[i, d * b:(d + 1) * b]))
            pv_365 = np.append(pv_365, np.sum(pv[i, d * b:(d + 1) * b]))
            hydro_365 = np.append(hydro_365, np.sum(hydro[d * b:(d + 1) * b]))
    demand_365 = demand_365.reshape(pro_num, Ltime)
    wind_factor_365 = wind_365.reshape(pro_num, Ltime)
    pv_factor_365 = pv_365.reshape(pro_num, Ltime)
    hydro_factor_365 = hydro_365.reshape(-1, Ltime)
    data_365 = np.concatenate((demand_365, wind_factor_365, pv_factor_365), axis=1)
    data_365 = np.concatenate((data_365, hydro_factor_365), axis=1)

    return data_365
