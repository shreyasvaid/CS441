import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import math

# Load df

CSV_PATH = "merged_stock_sentiment_filled.csv" 

df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH)
print("Columns:", list(df.columns))

# Ensure Date is in datetime and sort by Ticker + Date
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

sort_cols = []
if "Ticker" in df.columns:
    sort_cols.append("Ticker")
if "Date" in df.columns:
    sort_cols.append("Date")

if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)


# next_close_t = Close at the next row (per ticker)
# next_return_t = (next_close_t - Close_t) / Close_t

if not {"Close", "Ticker"}.issubset(df.columns):
    raise ValueError("Need 'Close' and 'Ticker' columns to build next_return target.")

df["next_close"] = df.groupby("Ticker")["Close"].shift(-1)
df["next_return"] = (df["next_close"] - df["Close"]) / df["Close"]

# Drop rows where next_return is NaN (last row per ticker)
df = df.dropna(subset=["next_return"]).copy()

target = "next_return"
print("Using target column:", target)

# Choose Features (all numeric except target)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude target and helper column next_close
features = [c for c in numeric_cols if c not in [target, "next_close"]]

print("Numeric columns:", numeric_cols)
print("Feature columns:", features)
print("Number of features:", len(features))

# Drop any remaining NaNs in features
df = df.dropna(subset=features + [target])

# Time-Ordered Train/Test Split
n = len(df)
split = int(n * 0.8)
train = df.iloc[:split]
test  = df.iloc[split:]

X_train = train[features].values
y_train = train[target].values
X_test  = test[features].values
y_test  = test[target].values

print(f"Train rows: {len(train)}, Test rows: {len(test)}")


# Scale Features, Fit OLS

scaler = StandardScaler()
# Scale features
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Add intercept/constant for statsmodels
X_train_sm = sm.add_constant(X_train_scaled, has_constant='add')
X_test_sm  = sm.add_constant(X_test_scaled,  has_constant='add')

# Fit OLS
model = sm.OLS(y_train, X_train_sm).fit()
print(model.summary())

# Use the same exog (with constant) for prediction
y_pred = model.predict(X_test_sm)

print(model.summary())


# Predictions & Metrics
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, np.nan))) * 100
r2 = r2_score(y_test, y_pred)

print("\n--- METRICS ---")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)
print("R²:", r2)

residuals = y_train - model.predict(X_train_sm)

sigma = residuals.std(ddof=1)
z = norm.ppf(0.95)  # 90% PI
half_width = z * sigma

pi_low  = y_pred - half_width
pi_high = y_pred + half_width


out_df = test.copy()
out_df["y_true"]    = y_test
out_df["y_pred"]    = y_pred
out_df["PI90_low"]  = pi_low
out_df["PI90_high"] = pi_high

out_df.to_csv("new_prediction_results.csv", index=False)
print("\nSaved: new_prediction_results.csv")


resid = y_test - y_pred

# Histogram
plt.figure()
plt.hist(resid, bins=30)
plt.title("Residual Histogram (New Dataset)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.savefig("new_residual_histogram.png", bbox_inches="tight")
plt.close()

# Residual vs Predicted
plt.figure()
plt.scatter(y_pred, resid, alpha=0.5)
plt.axhline(0, color="red")
plt.title("Residuals vs Predicted (New Dataset)")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.savefig("new_residual_vs_predicted.png", bbox_inches="tight")
plt.close()

print("Saved: new_residual_histogram.png")
print("Saved: new_residual_vs_predicted.png")

print("\nALL DONE")
