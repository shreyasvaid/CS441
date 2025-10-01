from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from datetime import datetime
import sqlite3
import os
import time, random

FINVIZ_URL = "https://finviz.com/quote.ashx?t="
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

DB_PATH = "finviz_news.db"

def fetch_news_table(ticker: str):
    url = FINVIZ_URL + ticker
    req = Request(url=url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urlopen(req)
    soup = BeautifulSoup(resp.read(), "html.parser")
    tbl = soup.find(id="news-table") or soup.find("table", class_="fullview-news-outer")
    return tbl

def parse_news_rows(ticker: str, news_table):
    """Return list[dict]: {ticker, day, time, title, source, link}"""
    rows = []
    current_date = None

    for tr in news_table.find_all("tr"):
        td_time = tr.find("td")
        if not td_time:
            continue
        date_text = td_time.get_text(strip=True).replace("Today", "").strip()
        a = tr.find("a")
        if a is None:
            continue

        title = a.get_text(strip=True)
        link = a.get("href", "")
        src_span = tr.find("span")
        source = src_span.get_text(strip=True).strip("()") if src_span else None

        # Case 1: full date like "Sep-28-25 09:15PM"
        if "-" in date_text and " " in date_text:
            date_part, time_part = date_text.split(" ", 1)
            current_date = datetime.strptime(date_part, "%b-%d-%y").strftime("%Y-%m-%d")
            time_str = time_part.strip()
        else:
            time_str = date_text
            if current_date is None:
                current_date = datetime.today().strftime("%Y-%m-%d")

        # Normalize time (24h vs 12h AM/PM)
        dt_obj = datetime.strptime(f"{current_date} {time_str}", "%Y-%m-%d %I:%M%p")
        day = dt_obj.strftime("%Y-%m-%d")
        time_only = dt_obj.strftime("%H:%M:%S")

        rows.append({
            "ticker": ticker,
            "day": day,
            "time": time_only,
            "title": title,
            "source": source,
            "link": link
        })
    return rows

def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS finviz_news (
            id     INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            day    TEXT NOT NULL,     -- 'YYYY-MM-DD'
            time   TEXT NOT NULL,     -- 'HH:MM:SS'
            title  TEXT NOT NULL,
            source TEXT,
            link   TEXT,
            UNIQUE (ticker, day, time, title)  -- dedup across ticker+day+time+title
        )
    """)
    conn.commit()

def upsert_rows(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany("""
        INSERT OR IGNORE INTO finviz_news (ticker, day, time, title, source, link)
        VALUES (:ticker, :day, :time, :title, :source, :link)
    """, rows)
    conn.commit()

def main():
    all_rows = []
    for t in TICKERS:
        print(f"Fetching {t}...")
        try:
            tbl = fetch_news_table(t)
            if not tbl:
                continue
            all_rows.extend(parse_news_rows(t, tbl))
        except Exception as e:
            print(f"Error fetching {t}: {e}")

        # sleep 1–3 seconds randomly to avoid 429 errors
        time.sleep(random.uniform(1, 3))

    with sqlite3.connect(DB_PATH) as conn:
        ensure_schema(conn)
        upsert_rows(conn, all_rows)

    print(f"Inserted (or skipped dups): {len(all_rows)} rows into {DB_PATH}")
    with sqlite3.connect(DB_PATH) as conn:
        for row in conn.execute("SELECT ticker, day, time, title FROM finviz_news ORDER BY day DESC, time DESC LIMIT 5"):
            print(row)

if __name__ == "__main__":
    main()
