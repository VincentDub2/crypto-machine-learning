from datetime import datetime, timedelta
import requests
from utils.date_utils import date_to_timestamp


def fetch_binance_data(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    all_data = []
    start_time_ts = date_to_timestamp(start_time)
    end_time_ts = date_to_timestamp(end_time)

    print(f"Fetching data from {datetime.utcfromtimestamp(start_time_ts / 1000)} to {datetime.utcfromtimestamp(end_time_ts / 1000)}")

    while start_time_ts < end_time_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time_ts,
            "endTime": min(start_time_ts + 1000 * 60 * 1000, end_time_ts), # 1000 points max
            'limit': 1000
        }

        print(f"Fetching data from {datetime.utcfromtimestamp(start_time_ts / 1000)} to {datetime.utcfromtimestamp(params['endTime'] / 1000)}")
        response = requests.get(url, params=params)

        data = response.json()
        print(f"Fetched {data} data points")

        if len(data) == 0:
            break

        print(f"Fetched {len(data)} data points")
        all_data.extend(data)
        print(f"Total data points fetched: {len(all_data)}")
        start_time_ts = data[-1][6]  # "Close time" de la dernière bougie

    print(f"Total data points fetched: {len(all_data)}")
    print(f"First data point: {all_data[0]}")
    print(f"Last data point: {all_data[-1]}")

    return all_data

