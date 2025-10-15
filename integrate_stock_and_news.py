"""
Integration script to merge sentiment-scored news (finviz_news_scored)
with hourly financial data (stock_prices_pct_chg)
and produce a unified table integrated_stock_news.

"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

NEWS_DB = Path("finviz_news.db")
PRICE_DB = Path("stock_history.db")
OUTPUT_DB = Path("integrated_stock_news.db")  # can be same as PRICE_DB if needed

#Load data
def load_data():
    with sqlite3.connect(NEWS_DB) as conn:
        news = pd.read_sql_query("SELECT ticker AS Ticker, day AS Date, time AS Time, sentiment_score FROM finviz_news_scored", conn)
    with sqlite3.connect(PRICE_DB) as conn:
        prices = pd.read_sql_query("SELECT * FROM stock_prices_pct_chg", conn)
    print(f"Loaded {len(news):,} news and {len(prices):,} price records.")
    return news, prices

# Preprocess and align timestamps
def preprocess(news, prices):
    # Ensure timestamp consistency
    news["DateTime"] = pd.to_datetime(news["Date"] + " " + news["Time"])
    prices["DateTime"] = pd.to_datetime(prices["Date"] + " " + prices["Time"])
    
    # Sort for merge-asof
    news = news.sort_values(["Ticker", "DateTime"])
    prices = prices.sort_values(["Ticker", "DateTime"])
    return news, prices

#Merge sentiment with prices
def integrate(news, prices):
    # We'll use merge_asof for time alignment (news <= price timestamp)
    merged = (
        pd.merge_asof(
            prices,
            news,
            by="Ticker",
            on="DateTime",
            direction="backward",  # only include news published at or before the hour
            tolerance=pd.Timedelta("24h")  # ignore stale news older than 1 day
        )
    )

    # Fill missing sentiment scores with 0 (neutral)
    merged["sentiment_score"] = merged["sentiment_score"].fillna(0.0)

    # Reorder + select key columns
    keep_cols = [
        "Ticker", "Date", "Time", "Open", "High", "Low", "Close",
        "Adj_Close", "Volume", "Pct_Change_Hourly", "Pct_Change_Daily",
        "sentiment_score"
    ]
    merged = merged[keep_cols].copy()

    print(f"Merged dataset shape: {merged.shape}")
    return merged

# Save integrated data
def save_to_sqlite(df):
    with sqlite3.connect(OUTPUT_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS integrated_stock_news (
                Ticker TEXT,
                Date TEXT,
                Time TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Adj_Close REAL,
                Volume INTEGER,
                Pct_Change_Hourly REAL,
                Pct_Change_Daily REAL,
                Sentiment_Score REAL,
                UNIQUE (Ticker, Date, Time)
            )
        """)
        df.to_sql("integrated_stock_news", conn, if_exists="replace", index=False)
    print(f"Saved integrated dataset to {OUTPUT_DB}")

# MAIN
def main():
    news, prices = load_data()
    news, prices = preprocess(news, prices)
    merged = integrate(news, prices)
    save_to_sqlite(merged)

if __name__ == "__main__":
    main()
