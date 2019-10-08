import csv
import pandas as pd
import io

def remove_csv_columns(path, new_path):
    with open(path, "r") as source:
        rdr = csv.reader(source)
        with open(new_path, "w") as result:
            wtr = csv.writer(result)
            for r in rdr:
                wtr.writerow((r[0], r[5], r[6]))

def interpolate_csv(path, new_path):
    z = pd.read_csv(path)
    z = z.set_index('Date')
    z.index = pd.to_datetime(z.index)
    z = z.resample('D').interpolate('linear')
    z.to_csv(new_path)

def process_csv(path, new_path):
    proc_df = []

    # re-evaluate stock growth
    with open(path, "r") as source:
        reader = csv.reader(source)
        row_num, last = 0, (0, 0, 0)
        for row in reader:
            if row_num > 1: 
                val_increase = (float(row[1]) - float(last[1])) / float(last[1])
                vol_increase = (float(row[2]) - float(last[2])) / float(last[2])
                proc_df.append((row[0], val_increase, vol_increase))
                last = row
            elif row_num == 1: last = row
            row_num += 1

    # parse max/min
    max_vol, min_vol = float('-inf'), float('inf')
    max_val, min_val = float('-inf'), float('inf')
    for row in proc_df:
        if row[1] > max_val: max_val = row[1]
        if row[1] < min_val: min_val = row[1]
        if row[2] > max_vol: max_vol = row[2]
        if row[2] < min_vol: min_vol = row[2]
    
    # re-scale stock growth
    new_proc_df = []
    for row in proc_df:
        val = (row[1] - min_val) / (max_val - min_val)
        vol = (row[2] - min_vol) / (max_vol - min_vol)
        new_proc_df.append((row[0], val, vol))

    # write final processed data
    with open(new_path, "w") as write_source:
        writer = csv.writer(write_source)
        writer.writerow(["Date", "Close", "Volume"])
        for row in new_proc_df:
            writer.writerow(row)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import dateutil.parser

data = np.genfromtxt(
    "bonsai/stock-data/SEAS-proc.csv", delimiter=',', names=True,
    dtype=None, converters={0: dateutil.parser.parse})

fig, ax = plt.subplots()
ax.plot(data['Date'], data['Close'], '-')
fig.autofmt_xdate()
plt.show()