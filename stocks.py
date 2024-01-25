import os
import polars as pl
import numpy as np
import itertools
import time
import seaborn as sb
import matplotlib.pyplot as plt

from typing import List, Dict
from collections import defaultdict
from datetime import timedelta, date
from polygon import RESTClient
from dotenv import load_dotenv
from polygon.rest.models import GroupedDailyAgg

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

client = RESTClient(api_key=POLYGON_API_KEY)


def fetch_data():
    prices = defaultdict(lambda: [])

    start_date = date(year=2023, month=1, day=2)  # first Monday is a holiday, so skip it

    # go through each day from 2023-01-02 to 2023-12-31 skipping Sat & Sun
    for day_offset in range(365):
    # for day_offset in range(5):
        day = start_date + timedelta(days=day_offset)

        if day.weekday() == 5 or day.weekday() == 6:
            continue

        ohlc_list: List[GroupedDailyAgg] = client.get_grouped_daily_aggs(date=day.strftime('%Y-%m-%d'), adjusted=False, include_otc=False)

        if len(ohlc_list) == 0:  # keep moving if the call failed
            continue

        print(f"Fetched {day.strftime('%A %Y-%m-%d')}")

        open = {str(ohlc.ticker).upper(): ohlc.open for ohlc in ohlc_list if ohlc.open is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        open["day"] = day.strftime('%Y-%m-%d')
        close = {str(ohlc.ticker).upper(): ohlc.close for ohlc in ohlc_list if ohlc.close is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        close["day"] = day.strftime('%Y-%m-%d')
        high = {str(ohlc.ticker).upper(): ohlc.high for ohlc in ohlc_list if ohlc.high is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        high["day"] = day.strftime('%Y-%m-%d')
        low = {str(ohlc.ticker).upper(): ohlc.low for ohlc in ohlc_list if ohlc.low is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        low["day"] = day.strftime('%Y-%m-%d')
        transactions = {str(ohlc.ticker).upper(): ohlc.transactions for ohlc in ohlc_list if ohlc.transactions is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        transactions["day"] = day.strftime('%Y-%m-%d')
        volume = {str(ohlc.ticker).upper(): ohlc.volume for ohlc in ohlc_list if ohlc.volume is not None and ohlc.ticker is not None and '.' not in ohlc.ticker}
        volume["day"] = day.strftime('%Y-%m-%d')

        prices['open'].append(open)
        prices['close'].append(close)
        prices['high'].append(high)
        prices['low'].append(low)
        prices['transactions'].append(transactions)
        prices['volume'].append(volume)

    pl.from_dicts(prices['open']).write_parquet('stock_prices/open.parquet', compression="snappy", statistics=True)
    print("Wrote open.parquet")

    pl.from_dicts(prices['close']).write_parquet('stock_prices/close.parquet', compression="snappy", statistics=True)
    print("Wrote close.parquet")

    pl.from_dicts(prices['high']).write_parquet('stock_prices/high.parquet', compression="snappy", statistics=True)
    print("Wrote high.parquet")

    pl.from_dicts(prices['low']).write_parquet('stock_prices/low.parquet', compression="snappy", statistics=True)
    print("Wrote low.parquet")

    pl.from_dicts(prices['transactions']).write_parquet('stock_prices/transactions.parquet', compression="snappy", statistics=True)
    print("Wrote transactions.parquet")

    pl.from_dicts(prices['volume']).write_parquet('stock_prices/volume.parquet', compression="snappy", statistics=True)
    print("Wrote volume.parquet")


if __name__ == "__main__":
    # fetch_data()
    # exit(1)

    df = pl.read_parquet('stock_prices/close.parquet')

    # remove any column (stock) where we don't have values for all days
    prices_df = df.select(col.name for col in df.null_count() if col.item() == 0).select(pl.exclude("day"))
    print(f'{len(prices_df.columns) - 1} stocks')

    cols = prices_df.columns
    N = len(prices_df)
    days = 5

    correlations = []

    count = 0
    start = time.monotonic()

    for s1, s2 in itertools.product(cols, cols):
        if s1 == s2:
            continue

        count += 1
        sub_cols = {f'{s2}+{day}': prices_df[s2].slice(day, N - days) for day in range(1, days+1)}
        sub_cols.update({s1: prices_df[s1].slice(0, N - days)})

        res = pl.DataFrame(sub_cols).corr()[s1]

        for i in range(days):
            if abs(res[i]) > 0.9:
                correlations.append([s1, s2, i+1, {res[1]}])
                # print(f'{s1} & {s2}+{i+1}: {res[i]:0.03f}')

        if len(correlations) == 50:
            break

        if count % 10_000 == 0:
            secs = time.monotonic() - start
            print(f"{count} in {secs:0.03f}s; {10_000.0 / (secs/60.0)} per minute")
            start = time.monotonic()

    dates = df["day"]

    for s1, s2, amt, corr in correlations:
        prices1 = prices_df[s1]
        prices2 = prices_df[s2]
        miss_count = 0
        total_count = len(prices1) - (amt+1)

        for i in range(total_count):
            change1 = prices1[i+1] - prices1[i]
            change2 = prices2[i+1+amt] - prices2[amt]

            if (change1 > 0 and change2 > 0) or (change1 < 0 and change2 < 0):
                # print(f"{dates[i]} - {dates[i+1]}: {prices1[i]} -> {prices1[i+1]}")
                # print(f"{dates[i+amt]} - {dates[i+1+amt]}: {prices2[i+amt]} -> {prices2[i+1+amt]}")
                pass
            else:
                miss_count += 1

        print(f"Missed {miss_count} of {total_count}; or wrong {miss_count / total_count:0.03f} - correlation: {corr}")
