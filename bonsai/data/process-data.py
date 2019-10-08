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

interpolate_csv("bonsai/stock-data/GSPC-col-proc.csv", "bonsai/stock-data/GSPC-inter-proc.csv")
interpolate_csv("bonsai/stock-data/SEAS-col-proc.csv", "bonsai/stock-data/SEAS-inter-proc.csv")