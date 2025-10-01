from datetime import datetime, timedelta
import sqlite3
import yfinance as yf
import pandas as pd

TICKERS = [
    # Technology
    'AAPL',   # Apple
    'MSFT',   # Microsoft
    'NVDA',   # NVIDIA
    'GOOGL',  # Alphabet (Google, Class A)
    'META',   # Meta Platforms (Facebook)
    
    # Consumer Discretionary
    'AMZN',   # Amazon
    'TSLA',   # Tesla
    
    # Financials
    'JPM',    # JPMorgan Chase
    'BAC',    # Bank of America
    'C',      # Citigroup
    'GS',     # Goldman Sachs
    'MS',     # Morgan Stanley
    
    # Healthcare
    'JNJ',    # Johnson & Johnson
    'UNH',    # UnitedHealth Group
    'PFE',    # Pfizer
    
    # Energy
    'XOM',    # ExxonMobil
    'CVX',    # Chevron
    
    # Industrials
    'CAT',    # Caterpillar
    'BA',     # Boeing

    # Communication Services
    'DIS',    # Disney
    'NFLX',   # Netflix

    # Major U.S. Index ETFs
    'SPY',    # S&P 500
    'QQQ',    # Nasdaq-100
    'DIA',    # Dow Jones Industrial Average
    'IWM'    # Russell 2000
]

DB_PATH = "stock_history.db"

def get_long_history(
    tickers,
    start_date=(datetime.today().date() - timedelta(days=365)), 
    end_date=datetime.today().date(),
    interval="1h",
    group_by="ticker",
    tz_local="America/New_York",
):
    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        actions=True,
        auto_adjust=False,
        back_adjust=False,
        interval=interval,
        group_by=group_by,
        threads=True,
        progress=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(0, future_stack=True).reset_index()
        if "level_1" in df.columns:
            df = df.rename(columns={"level_1": "Ticker"})
        if "Datetime" not in df.columns:
            dtcol = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            if dtcol:
                df = df.rename(columns={dtcol[0]: "Datetime"})
    else:
        df = df.reset_index()
        if "Datetime" not in df.columns:
            df = df.rename(columns={"Date": "Datetime"})
        df["Ticker"] = tickers[0] if isinstance(tickers, list) and len(tickers) == 1 else "TICKER"

    if pd.api.types.is_datetime64_any_dtype(df["Datetime"]):
        if getattr(df["Datetime"].dt, "tz", None) is None:
            df["Datetime"] = df["Datetime"].dt.tz_localize("UTC")
        df["Datetime_local"] = df["Datetime"].dt.tz_convert(tz_local)
    else:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df["Datetime_local"] = df["Datetime"].dt.tz_convert(tz_local)

    # Split into Date (YYYY-MM-DD) and Time (HH:MM 24h)
    df["Date"] = df["Datetime_local"].dt.strftime("%Y-%m-%d")
    df["Time"] = df["Datetime_local"].dt.strftime("%H:%M:%S")

    # Normalize column names to match DB schema
    df = df.rename(columns={
        "Adj Close": "Adj_Close",
        "Stock Splits": "Stock_Splits",
    })

    # Keep only the fields we store
    keep_cols = [
        "Date", "Time", "Ticker",
        "Open", "High", "Low", "Close", "Adj_Close",
        "Volume", "Dividends", "Stock_Splits"
    ]
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    df = df[keep_cols].copy()

    # Sort for readability
    df = df.sort_values(["Ticker", "Date", "Time"], kind="mergesort").reset_index(drop=True)

    print(df.head())
    print(df.tail())
    return df


def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date         TEXT NOT NULL,
            Time         TEXT NOT NULL,
            Ticker       TEXT NOT NULL,
            Open         REAL,
            High         REAL,
            Low          REAL,
            Close        REAL,
            Adj_Close    REAL,
            Volume       INTEGER,
            Dividends    REAL,
            Stock_Splits REAL,
            UNIQUE(Date, Time, Ticker)  -- hourly uniqueness
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_datetime ON stock_prices (Ticker, Date, Time)")
    conn.commit()


def save_to_sqlite(df: pd.DataFrame, db_path=DB_PATH, batch_size=1000):
    required = ["Date","Time","Ticker","Open","High","Low","Close","Adj_Close","Volume","Dividends","Stock_Splits"]
    rows = list(df[required].itertuples(index=False, name=None))

    insert_sql = """
        INSERT OR IGNORE INTO stock_prices
        (Date, Time, Ticker, Open, High, Low, Close, Adj_Close, Volume, Dividends, Stock_Splits)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    with sqlite3.connect(db_path) as conn:
        ensure_schema(conn)
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            conn.executemany(insert_sql, batch)
            conn.commit()
            print(f"Inserted batch {i//batch_size + 1} ({len(batch)} rows)")


def main():
    df = get_long_history(
        tickers=TICKERS,
    )
    save_to_sqlite(df)

    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM stock_prices").fetchone()[0]
        print("Total rows in DB:", count)


if __name__ == "__main__":
    main()
