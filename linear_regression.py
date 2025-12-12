import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import math

OUTPUT_DIR = Path("phase4_linear_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# load data
# Option A: From SQLite:
# df = pd.read_sql("SELECT * FROM integrated_stock_news", sqlite3.connect("integrated_stock_news.db"))

# Option B: From CSV (your uploaded dataset):
df = pd.read_csv("stock_sentiment_data.csv")

# Ensure correct typing
df = df.sort_values(["ticker","date"]).reset_index(drop=True)

# define features
features = ["avg_sentiment","total_sentiment","max_sentiment","min_sentiment",
            "num_articles","Close","return_prev_to_close"]
target = "return_close_to_next"

df = df.dropna(subset=features + [target]).copy()

#train test
split = int(len(df)*0.8)
train = df.iloc[:split]
test  = df.iloc[split:]

X_train = train[features].astype(float)
y_train = train[target].astype(float)
X_test  = test[features].astype(float)
y_test  = test[target].astype(float)

#scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

X_train_s = sm.add_constant(X_train_s)
X_test_s  = sm.add_constant(X_test_s)

model = sm.OLS(y_train, X_train_s).fit()

# metrics
y_pred = model.predict(X_test_s)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# pred interval
sigma = np.std(y_train - model.predict(X_train_s))
z = norm.ppf(0.95)
half_width = z * sigma

pi_lower = y_pred - half_width
pi_upper = y_pred + half_width

pred_df = test[["ticker","date"]].copy()
pred_df["y_true"]  = y_test.values
pred_df["y_pred"]  = y_pred
pred_df["PI90_low"]  = pi_lower
pred_df["PI90_high"] = pi_upper
pred_df.to_csv(OUTPUT_DIR/"prediction_results.csv", index=False)

# residual
residuals = y_test - y_pred

# Residual Histogram
plt.figure()
plt.hist(residuals, bins=30)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.savefig(OUTPUT_DIR/"residual_histogram.png")
plt.close()

# Residual vs Predicted
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted Return")
plt.ylabel("Residual")
plt.savefig(OUTPUT_DIR/"residual_vs_predicted.png")
plt.close()

example = pred_df[pred_df["ticker"] == pred_df["ticker"].iloc[0]].tail(50)
plt.figure()
plt.plot(example["date"], example["y_true"], label="Actual", linewidth=2)
plt.plot(example["date"], example["y_pred"], label="Predicted", linestyle="--")
plt.fill_between(example["date"], example["PI90_low"], example["PI90_high"], alpha=0.2)
plt.xticks(rotation=45)
plt.title("Predicted vs Actual w/ 90% Interval")
plt.savefig(OUTPUT_DIR/"forecast_band.png")
plt.close()

print("MODEL PERFORMANCE")
print(f"R²: {r2:.4f}")
print(f"RMSE: {rmse:.6f}")
print(f"Residual Std Dev (σ): {sigma:.6f}")
print(f"90% Prediction Interval Half-Width: {half_width:.6f}")
print("\n📁 All outputs saved to:", OUTPUT_DIR.absolute())
