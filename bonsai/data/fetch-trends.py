import datetime
import pytrends
import csv
import matplotlib.pyplot as plt
from pytrends.request import TrendReq

def last_day_of_month(any_day):
    next_month = any_day.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)

def parse_phrase_trend(phrase, location='US', filter=''):

    # search trend data
    print("Querying full trend data.")
    pytrend = TrendReq(hl='en-US')
    pytrend.build_payload([phrase], timeframe='2013-05-01 2019-09-01', geo=location, gprop=filter)
    interest_df = pytrend.interest_over_time()
    full_df = [row[0] for _, row in interest_df.iterrows()]
    print("Done querying full trend data.\n")

    # fetch monthly data
    monthly_dfs = []
    current_date = datetime.datetime(year=2013, month=5, day=1)
    for i in range(len(full_df)):
        print("Querying trend data for month {}.".format(i + 1))

        # query current range
        pytrend = TrendReq(hl='en-US')
        curr_range = current_date.strftime("%Y-%m-%d") + " " + last_day_of_month(current_date).strftime("%Y-%m-%d")
        pytrend.build_payload([phrase], timeframe=curr_range, geo=location, gprop=filter)
        interest_df = pytrend.interest_over_time()
        monthly_dfs.append([row[0] for _, row in interest_df.iterrows()])

        # determine next start date
        current_date += datetime.timedelta(weeks=5)
        current_date = current_date.replace(day=1)
    print("Done querying monthly trend data.\n")

    # normalize monthly data
    max_dp, min_dp = -1, 101
    normalized_df = []
    for i in range(len(monthly_dfs)):
        print("Normalizing trend data for month {}.".format(i))
        for data in monthly_dfs[i]:
            new_dp = data * (full_df[i] / 100.0)
            normalized_df.append(new_dp)
            if new_dp > max_dp: max_dp = new_dp
            if new_dp < min_dp: min_dp = new_dp
    print("Done normalizing monthly trend data.\n")

    # normalize over entire dataset
    print("Normalizing full trend data.")
    final_data = [(100.0 / (max_dp - min_dp)) * (x - max_dp) + max_dp for x in normalized_df]
    print("Done normalizing full trend data.\n")
    return final_data

def create_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        current_date = datetime.datetime(year=2013, month=5, day=1)

        # write header data
        fieldnames = ['Date', 'Popularity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # write row data
        for field in data:
            writer.writerow({
                'Date': current_date.strftime("%Y-%m-%d"),
                'Popularity': field / 100.0
            })
            current_date += datetime.timedelta(days=1)

if __name__ == "__main__":
    data = parse_phrase_trend("SeaWorld")
    create_csv(data, "bonsai\\trend-data\\trend-seaworld.csv")
    plt.plot(data)
    plt.show()