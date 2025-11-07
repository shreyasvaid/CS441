import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# variables
PATH = "stock_sentiment_data_time.csv" # modified with compare_ts.py
TICKER = "AAPL"
TEST_SPLIT = 0.1  # last 10% of trading days for testing
HIST_DAYS = 183  # how many days before test set to include in plot

# prep data
df = pd.read_csv(PATH, parse_dates=["datetime"])
df_ticker = df[df["ticker"] == TICKER].copy().sort_values("datetime")
df_ticker["time"] = df_ticker["datetime"].dt.time
df_close_daily = df_ticker[df_ticker["time"] == pd.to_datetime("15:30", format="%H:%M").time()].copy()
df_close_daily = df_close_daily.set_index("datetime").sort_index()

ts = df_close_daily["Close"]
ts_log = np.log(ts)
test_days = int(len(ts) * TEST_SPLIT)

sent_cols = ["num_articles", "avg_sentiment"]#, "total_sentiment", "max_sentiment", "min_sentiment"]
exog = df_close_daily.get(sent_cols, pd.DataFrame(index=df_close_daily.index))
exog = exog.fillna(0)

# Train/Test split
train_ts = ts_log.iloc[:-test_days]
test_ts = ts_log.iloc[-test_days:]

train_exog = exog.iloc[:-test_days]
test_exog = exog.iloc[-test_days:]

# find best (p,d,q)
auto_model_no_exog = pm.auto_arima(train_ts, 
                                   seasonal=False, 
                                   stepwise=True,
                                   suppress_warnings=True, 
                                   error_action='ignore',
                                   trace=True)
P1, D1, Q1 = auto_model_no_exog.order

auto_model_with_exog = pm.auto_arima(train_ts, 
                                     exogenous=train_exog,
                                     seasonal=False, 
                                     stepwise=True,
                                     suppress_warnings=True, 
                                     error_action='ignore',
                                     trace=True)
P2, D2, Q2 = auto_model_with_exog.order

print(f"Best order without sentiment: (p={P1}, d={D1}, q={Q1})")
print(f"Best order with sentiment:    (p={P2}, d={D2}, q={Q2})")

# fit models
model_no_exog = SARIMAX(train_ts, 
                        order=(P1,D1,Q1),
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
fit_no_exog = model_no_exog.fit(disp=False)

model_with_exog = SARIMAX(train_ts, 
                          exog=train_exog, 
                          order=(P2,D2,Q2),
                          enforce_stationarity=False, 
                          enforce_invertibility=False)
fit_with_exog = model_with_exog.fit(disp=False)

# forecast test period
forecast_no_exog = np.exp(fit_no_exog.get_forecast(steps=test_days).predicted_mean)
forecast_with_exog = np.exp(fit_with_exog.get_forecast(steps=test_days, exog=test_exog).predicted_mean)

# Convert actual test data back from log
actual = np.exp(test_ts)
forecast_no_exog.index = actual.index
forecast_with_exog.index = actual.index

# Compute error metrics
def compute_errors(pred, actual):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(actual, pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }

errors_no_exog = compute_errors(forecast_no_exog, actual)
errors_with_exog = compute_errors(forecast_with_exog, actual)

print("\nModel Comparison on Test Set:")
print("Without Sentiment:", errors_no_exog)
print("With Sentiment:   ", errors_with_exog)

if errors_with_exog["RMSE"] < errors_no_exog["RMSE"]:
    winner = "Model WITH sentiment is better!"
else:
    winner = "Model WITHOUT sentiment is better!"

print("\nDecision:", winner)

# Plot actual vs predictions
plot_start = max(0, len(ts) - test_days - HIST_DAYS)
plot_series = np.exp(ts_log.iloc[plot_start:])  # back-transform from log

# Forecast test period
forecast_no_exog.index = actual.index
forecast_with_exog.index = actual.index

plt.figure(figsize=(12, 6))
plt.plot(plot_series.index, plot_series.values, label="Historical", color="blue")
plt.plot(forecast_no_exog.index, forecast_no_exog.values, label="Forecast (No Sentiment)", color="red")
plt.plot(forecast_with_exog.index, forecast_with_exog.values, label="Forecast (With Sentiment)", color="green")
plt.title(f"{TICKER} - Historical vs Forecast")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
