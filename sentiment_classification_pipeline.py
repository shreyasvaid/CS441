import sqlite3
import pandas as pd
import re
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import torch

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def connect_db(db):
    """Connect to the database and get the news data as a pandas df"""
    # Use a raw string (r"") or forward slashes ("/") for the file path
    conn = sqlite3.connect(db)

    # Read the table into a DataFrame
    df = pd.read_sql_query("SELECT * FROM finviz_news", conn)

    # Close the connection
    conn.close()
    
    # return the dataframe
    return df

def clean_text(s:str)->str:
    """Cleans up strings: handle missing values, removes whitespaces, removes URL"""
    if pd.isna(s):
        return ""
    s = s.strip()
    s = re.sub(r"http\S+|www\.\S+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def compose_text(row, include_source=True):
    """Builds out the text to be scored by combining news headline with 
        its source in case of added signal/bias from source"""
    title = clean_text(row.get("title", ""))
    if include_source:
        src = clean_text(row.get("source",""))
        return f"{title} - {src}" if src else title
    return title

def finbert_scores(texts):
    """Scores sentiment of the title + source (if it provides) on a scale from [-1,1]"""
    model_name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True)
    
    scores=[]
    bs=32
    for i in range(0,len(texts),bs):
        batch = texts[i:i+bs]
        outs=pipe(batch) #batch processing for the data
        for dist in outs:
            d={x["label"].lower(): x["score"] for x in dist}
            score = float(d.get("positive", 0.0) - d.get("negative", 0.0))
            scores.append(score)
    return np.array(scores, dtype=float)

def vader_scores(texts):
    """Vader model in case of failure from the finbert model"""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(t)["compound"] for t in texts], dtype=float)

def add_sentiment(df:pd.DataFrame, include_source_in_text=True, source_weights:dict | None = None):
    """Takes in the news data, makes a copy, cleans the text and puts the title and source together, then scores the sentiment of each."""
    out = df.copy()
    texts = out.apply(lambda r: compose_text(r, include_source=include_source_in_text), axis=1).tolist()
    try:
        scores = finbert_scores(texts)
    except Exception:
        # Runs vader scoring if finbert fails
        scores = vader_scores(texts)
    
    out["sentiment_score"] = scores
    
    # If different news sources weights are provided (bias), factor into the model
    if source_weights:
        w = out["source"].map(source_weights).fillna(1.0)
        out["sentiment_score"] = out["sentiment_score"] * w
    return out

import sqlite3

def upsert_scored(df_scored, db, table="finviz_news_scored"):
    cols = ["id","ticker","day","time","title","source","link","sentiment_score"]
    rows = list(df_scored[cols].itertuples(index=False, name=None))

    with sqlite3.connect(db) as conn:
        # 1) Ensure table exists with a primary key on id
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                day TEXT,
                time TEXT,
                title TEXT,
                source TEXT,
                link TEXT,
                sentiment_score REAL
            )
        """)

        # 2) Upsert rows
        conn.executemany(f"""
            INSERT INTO {table} (id, ticker, day, time, title, source, link, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                ticker = excluded.ticker,
                day = excluded.day,
                time = excluded.time,
                title = excluded.title,
                source = excluded.source,
                link = excluded.link,
                sentiment_score = excluded.sentiment_score
        """, rows)
        conn.commit()
    
def main():
    db = "finviz_news.db"
    
    # Connect to db
    df = connect_db(db)

    # Scores sentiment for the data
    df_scored = add_sentiment(df, include_source_in_text=True)
    
    # Scores sentiment for the data with giving news sources specific weights (no bias is a value of 1)
    # df_scored = add_sentiment(df, include_source_in_text=True, source_weights={"CNBC TV": 1.05, "Insider Monkey": 0.95})
    
    upsert_scored(df_scored, db)
    
    print(df_scored.head())    # has `sentiment_score
    
    
if __name__ == "__main__":
    
    main()