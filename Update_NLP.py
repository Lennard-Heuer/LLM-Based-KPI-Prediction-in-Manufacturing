import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openai
import re
import json
import matplotlib.patches as patches
import ast
import matplotlib.gridspec as gridspec
import seaborn as sns
import textwrap

print("py script started")

CT_WS1 = current.CT_WS1.value
CT_WS2 = current.CT_WS2.value
CT_WS3 = current.CT_WS3.value

buffer1freeportion = current.Buffer1.StatEmptyPortion
arr_buffer2freeportion = current.ArrivalBuffer2.StatEmptyPortion
buffer2freeportion = current.Buffer2.StatEmptyPortion
arr_buffer3freeportion = current.ArrivalBuffer3.StatEmptyPortion


cache_buffer1_cap = int(current.Buffer1Cap.value)
cache_arr_buffer2_cap = int(current.Arr_Buffer2Cap.value)
cache_buffer2_cap = int(current.Buffer2Cap.value)
cache_arr_buffer3_cap = int(current.Arr_Buffer3Cap.value)

Workstation1_Availability = float(current.WSAv1.value)
Workstation2_Availability = float(current.WSAv2.value)
Workstation3_Availability = float(current.WSAv3.value)

AGVNo = int(current.VehicleNo.value)
AVG_Util = float(current.AGV_Util.value)
#AGV_Util = float(current.MUs.Transporter.StatWorkTimePortion)

#print( f"""MUMUMUMU {AGV_Util}""")

####################################################################################
# CALCUALTION OF STANDART FACTORY PHYSICS PARAMETERS
####################################################################################

# Factory Physics textbook parameters (converted to hours/jobs/hour)
# All static, based on your textbook input
mean_proc_1 = 0.00556  # h
mean_proc_2 = 0.03889  # h
mean_proc_3 = 0.00625  # h

machines_1 = 3
machines_2 = 2
machines_3 = 3

capacity_1 = (machines_1 / mean_proc_1) * (Workstation1_Availability / 100)   # jobs/hour
capacity_2 = (machines_2 / mean_proc_2) * (Workstation2_Availability / 100)   # jobs/hour (bottleneck)
capacity_3 = (machines_3 / mean_proc_3) * (Workstation3_Availability / 100)   # jobs/hour

bottleneck_rate = min(capacity_1, capacity_2, capacity_3) 
T0 = mean_proc_1 / (Workstation1_Availability / 100) + mean_proc_2 / (Workstation2_Availability / 100) + mean_proc_3 / (Workstation3_Availability / 100)  # 0.0507 h
W0 = bottleneck_rate * T0                                 
TH_worst = 1 / T0                                          


def best_case_CT(wip, T0, W0, r_b):
    # Returns best-case cycle time in hours
    if wip <= W0:
        return T0
    else:
        return wip / r_b

def best_case_TH(wip, T0, W0, r_b):
    # Returns best-case throughput in jobs/hour
    if wip <= W0:
        return wip / T0
    else:
        return r_b

def worst_case_CT(wip, T0):
    return wip * T0

def worst_case_TH(T0):
    return 1 / T0

def practical_worst_case_CT(wip, T0, rb):
    return T0 + (wip - 1)/rb

def practical_worst_case_TH(wip, T0, rb):
    return wip / (T0 + (wip - 1)/rb)


import numpy as np



util1 = current.Utilization_Station_1.value
util2 = current.Utilization_Station_2.value
util3 = current.Utilization_Station_3.value




'''
def propagate_once_ws2_ws3_ws1(u_sq, c_e_sq, num_servers):
    """
    Propagate SCV for a single loop: WS2 → WS3 → WS1.
    u_sq: [u2^2, u3^2, u1^2]
    c_e_sq: [c_e2^2, c_e3^2, c_e1^2]
    num_servers: [m2, m3, m1]
    Returns: [c_d2^2, c_d3^2, c_d1^2]
    """
    # Step 1: WS2 (bottleneck)
    c_a2_sq = 0  # perfectly regular arrival
    c_e2_sq = c_e_sq[0]
    u2_sq = u_sq[0]
    m2 = num_servers[0]
    cd2_sq = 1 + (1 - u2_sq) * (c_a2_sq - 1) + u2_sq * (1 / np.sqrt(m2)) * (c_e2_sq - 1)

    # Step 2: WS3
    c_a3_sq = cd2_sq
    c_e3_sq = c_e_sq[1]
    u3_sq = u_sq[1]
    m3 = num_servers[1]
    cd3_sq = 1 + (1 - u3_sq) * (c_a3_sq - 1) + u3_sq * (1 / np.sqrt(m3)) * (c_e3_sq - 1)

    # Step 3: WS1
    c_a1_sq = cd3_sq
    c_e1_sq = c_e_sq[2]
    u1_sq = u_sq[2]
    m1 = num_servers[2]
    cd1_sq = 1 + (1 - u1_sq) * (c_a1_sq - 1) + u1_sq * (1 / np.sqrt(m1)) * (c_e1_sq - 1)

    return cd2_sq, cd3_sq, cd1_sq

# === Use your example values ===
u_sq =    [0.4096, 0.0196, 0.09]   # [u2^2, u3^2, u1^2]
c_e_sq =  [1.0,    0.10,   0.063]  # [c_e2^2, c_e3^2, c_e1^2]
m =       [1,      3,      3]      # [m2, m3, m1]

cd2_sq, cd3_sq, cd1_sq = propagate_once_ws2_ws3_ws1(u_sq, c_e_sq, m)'''



calcuations = f"""
# Factory Physics textbook parameters (converted to hours/jobs/hour)
# All static, based on your textbook input
mean_proc_1 = 0.00556  # h
mean_proc_2 = 0.03889  # h
mean_proc_3 = 0.00625  # h

machines_1 = 3
machines_2 = 2
machines_3 = 3

capacity_1 = (machines_1 / mean_proc_1) * (Workstation1_Availability / 100)   # jobs/hour
capacity_2 = (machines_2 / mean_proc_2) * (Workstation2_Availability / 100)   # jobs/hour (bottleneck)
capacity_3 = (machines_3 / mean_proc_3) * (Workstation3_Availability / 100)   # jobs/hour

bottleneck_rate = min(capacity_1, capacity_2, capacity_3) 
T0 = mean_proc_1 / (Workstation1_Availability / 100) + mean_proc_2 / (Workstation2_Availability / 100) + mean_proc_3 / (Workstation3_Availability / 100)  # 0.0507 h
W0 = bottleneck_rate * T0                                 
TH_worst = 1 / T0                                          


def best_case_CT(wip, T0, W0, r_b):
    # Returns best-case cycle time in hours
    if wip <= W0:
        return T0
    else:
        return wip / r_b

def best_case_TH(wip, T0, W0, r_b):
    # Returns best-case throughput in jobs/hour
    if wip <= W0:
        return wip / T0
    else:
        return r_b

def worst_case_CT(wip, T0):
    return wip * T0

def worst_case_TH(T0):
    return 1 / T0

def practical_worst_case_CT(wip, T0, rb):
    return T0 + (wip - 1)/rb

def practical_worst_case_TH(wip, T0, rb):
    return wip / (T0 + (wip - 1)/rb)
"""


#####################################################################################


NLP_description = f"""
General description of the system:

3 Stations in one line, AGVs connecting them
Station1 : 3 stations, Processing Time / Distribution: Normal, mu20sec, sigma5sec, lowerbound 0sec upperbound 45sec
Station2: 2 stations, Processing Time / Distribution: Negexp., 2min20sec for beta
Station3: 3 stations, Processing Time / Distribution: Uniform, lowerbound 10sec upperbound 35sec

There are Buffers between the stations:
Buffer1 between Parallelstation1 and the AGV transport
ArrivalBuffer2 between the AGV transport and Parallelstation2
Buffer2 between Parallelstation2 and the AGV transport
ArrivalBuffer3 between the AGV transport and Parallelstation3

One AGV Pool with 5 AGVs, the pool managing the transportation system between the stations
This is the Capacity of Buffer1: {cache_buffer1_cap} MUs
This is the Capacity of ArrivalBuffer2: {cache_arr_buffer2_cap} MUs
This is the Capacity of Buffer2: {cache_buffer2_cap} MUs
This is the Capacity of ArrivalBuffer3: {cache_arr_buffer3_cap} MUs

This is the availability of Workstation1: {Workstation1_Availability}%
This is the availability of Workstation2: {Workstation2_Availability}%
This is the availability of Workstation3: {Workstation3_Availability}%

Number of AGVs in operation: {AGVNo}
"""


current.NLP_D.value = NLP_description