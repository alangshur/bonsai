import numpy as np
import csv

def process_model_data():

    # gather SEAS data
    SEAS_data = []
    with open("bonsai/stock-data/SEAS-proc.csv") as r_file:
        reader = csv.reader(r_file)
        next(reader)
        for row in reader:
            SEAS_data.append([row[1], row[2]])

    # gather GSPC data
    GSPC_data = []
    with open("bonsai/stock-data/GSPC-proc.csv") as r_file:
        reader = csv.reader(r_file)
        next(reader)
        for row in reader:
            GSPC_data.append([row[1], row[2]])

    # gather trend data
    trend_data = []
    with open("bonsai/trend-data/trend-seaworld.csv") as r_file:
        reader = csv.reader(r_file)
        next(reader)
        for row in reader:
            trend_data.append([row[1]])

    # combine data
    combined_data = []
    for i in range(len(trend_data)):
        combined_data.append(SEAS_data[i] + GSPC_data[i] + trend_data[i])
    return np.array(combined_data)

print(process_model_data())