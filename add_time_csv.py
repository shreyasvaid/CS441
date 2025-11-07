import pandas as pd

df = pd.read_csv("stock_sentiment_data.csv")

# Ensure date column is parsed correctly
df["date"] = pd.to_datetime(df["date"])

# Define trading hours (9:30 to 3:30)
times = pd.date_range("09:30", "15:30", freq="1H").time  # hourly increments

# Function to assign times to each group
def assign_times(group):
    n = len(group)
    # Repeat or truncate times to match the number of rows
    assigned_times = [times[i % len(times)] for i in range(n)]
    group = group.copy()
    group["time"] = assigned_times
    return group

# Apply the function by ticker and date
df = df.groupby(["ticker", "date"], group_keys=False).apply(assign_times)

# Combine date and time into a single datetime if you want
df["datetime"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"].astype(str))

# Save to a new CSV
df.to_csv("stock_sentiment_data_time.csv", index=False)

print(df.head(15))
