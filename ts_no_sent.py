import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# variables
PATH = "stock_sentiment_data_time.csv" # modified with compare_ts.py
TICKER = "AAPL"
ALPHA = 0.20
NUM_FORECASTED_DAYS = 5

# prep data
df = pd.read_csv(PATH, parse_dates=["datetime"])
df_ticker = df[df["ticker"] == TICKER].copy().sort_values("datetime")
df_ticker["time"] = df_ticker["datetime"].dt.time
df_close_daily = df_ticker[df_ticker["time"] == pd.to_datetime("15:30", format="%H:%M").time()].copy()

ts = df_close_daily.set_index("datetime")["Close"]
ts_log = np.log(ts)  # log-transform

# find best (p,d,q)
auto_model = pm.auto_arima(ts_log,
                           seasonal=False,
                           stepwise=True,
                           suppress_warnings=True,
                           error_action='ignore',
                           trace=True)
P, D, Q = auto_model.order

# fit ARIMA model
model = SARIMAX(ts_log, 
                order=(P, D, Q),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# forecast next 5 trading days
forecast_res = model_fit.get_forecast(steps=NUM_FORECASTED_DAYS)

# back-transform from log
forecast_mean = np.exp(forecast_res.predicted_mean)
conf_int = np.exp(forecast_res.conf_int(alpha=ALPHA))

# define forecast dates
last_date = ts.index[-1]
forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=NUM_FORECASTED_DAYS)
forecast_dates = [d.replace(hour=15, minute=30) for d in forecast_dates]
forecast_mean.index = forecast_dates
conf_int.index = forecast_dates

# display forecast
forecast_df = pd.DataFrame({
    "forecast": forecast_mean,
    f"lower_{100*(1-ALPHA)}_PI": conf_int.iloc[:, 0],
    f"upper_{100*(1-ALPHA)}_PI": conf_int.iloc[:, 1]
})
print(f"\n{TICKER} Forecast (3:30 PM for next 5 trading days):")
print(forecast_df)

# diagnostics
print(model_fit.summary())

# plot data and forecast PI
plt.figure(figsize=(12, 6))
plt.plot(ts, label="Historical 3:30 PM Close", color="blue")
plt.plot(forecast_mean.index, forecast_mean.values, color="red", label="Forecast (3:30 PM Close)")
plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=ALPHA, label=f"{100*(1-ALPHA)}% PI")
plt.title(f"{TICKER} - 3:30 PM Close Forecast (Next 5 Trading Days)\n(NOT Including Sentiment Features)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()