# Stock Return Prediction Using News Sentiment  
**CS 441 – Data Mining Capstone Project**

## Project Description
This project explores whether financial news sentiment can be used to predict short-term stock returns. Using publicly available news headlines and historical market data, we construct an end-to-end data pipeline that collects, processes, and aligns sentiment information with stock price movements. The objective of this project is not to claim market-beating performance, but rather to evaluate whether sentiment signals contain measurable predictive information when applied to real-world financial data.

## Data Sources
News data is collected from FinViz for a selected set of U.S. stocks and ETFs. Each headline is timestamped, deduplicated, and stored in a local SQLite database. Sentiment analysis is applied to every headline using a financial-domain NLP model, producing positive, neutral, negative, and aggregate sentiment scores. These sentiment values are later used as input features for modeling.

Historical stock price data is retrieved from Yahoo Finance. The price dataset includes Open, High, Low, Close, Adjusted Close, Volume, Dividends, and Stock Splits. From this data, returns are computed and used as prediction targets. All price data is aligned with sentiment data using timestamps to ensure that no future information is leaked into the training process.

## Data Processing Pipeline
The project follows a structured data processing pipeline. News headlines are scraped and sentiment-scored, while historical stock prices are downloaded separately. The two data sources are then temporally aligned by ticker and date. Missing sentiment values are handled through forward-filling to preserve continuity. Feature engineering is applied to construct numerical inputs and a next-period return target. Finally, the dataset is split chronologically into training and testing sets to reflect realistic forecasting conditions.

All intermediate datasets are stored locally using SQLite to ensure reproducibility and transparency.

## Modeling Overview
This project implements three modeling approaches: time series models, linear regression, and CatBoost. Each model is trained and evaluated using the same processed dataset and consistent train-test splits to allow for fair comparison.

## Evaluation
Model performance is evaluated using standard regression metrics, including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R². In addition to numerical metrics, residual diagnostics and visualizations are used to better understand model behavior and error structure.

## Key Takeaways
Short-term stock returns are highly noisy and difficult to predict, even when incorporating sentiment data. Increasing dataset size and feature richness does not necessarily lead to improved predictive performance. The project highlights the importance of careful data alignment, realistic evaluation methods, and transparent interpretation of results when working with financial time-series data.

## Technologies Used
Python is used as the primary programming language. Core libraries include pandas, NumPy, statsmodels, scikit-learn, transformers (for FinBERT), NLTK, yfinance, SQLite, and Matplotlib.

## Academic Context
This project was completed as a capstone for CS 441 (Data Mining) and demonstrates applied skills in data collection, preprocessing, modeling, evaluation, and interpretation using real-world financial data.
