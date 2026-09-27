from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "retail_store_inventory.csv"
PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "processed_inventory_data.csv"
GROUPS = ["Store ID", "Product ID"]
FEATURES = ["current_stock", "sales_velocity", "lead_time_days", "demand_variability", "promotion_status", "price", "discount", "category", "region", "weather_condition", "seasonality", "day_of_week", "month"]
NUMERIC_FEATURES = ["current_stock", "sales_velocity", "lead_time_days", "demand_variability", "promotion_status", "price", "discount", "day_of_week", "month"]
CATEGORICAL_FEATURES = ["category", "region", "weather_condition", "seasonality"]

def build_dataset(raw_path=RAW, save_path=PROCESSED):
    df = pd.read_csv(raw_path, parse_dates=["Date"])
    df = df.sort_values(GROUPS + ["Date"]).reset_index(drop=True)
    g = df.groupby(GROUPS, sort=False)
    # Use only prior observations for derived predictors.
    df["sales_velocity"] = g["Units Sold"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["demand_variability"] = g["Units Sold"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())
    df["sales_velocity"] = df["sales_velocity"].fillna(df["Units Sold"].median())
    df["demand_variability"] = df["demand_variability"].fillna(df["Units Sold"].std()).clip(lower=0)
    # Source does not contain supplier lead time. Keep this explicit planning assumption.
    df["lead_time_days"] = 3.0
    df["current_stock"] = df["Inventory Level"]
    df["promotion_status"] = df["Holiday/Promotion"]
    df["price"] = df["Price"]
    df["discount"] = df["Discount"]
    df["category"] = df["Category"]
    df["region"] = df["Region"]
    df["weather_condition"] = df["Weather Condition"]
    df["seasonality"] = df["Seasonality"]
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    # Derived label: any future day in the next 7 days where inventory is no greater than observed sales.
    future_events = []
    for horizon in range(1, 8):
        future_inventory = g["Inventory Level"].shift(-horizon)
        future_sales = g["Units Sold"].shift(-horizon)
        future_events.append((future_inventory <= future_sales).fillna(False))
    df["stockout_within_7_days"] = pd.concat(future_events, axis=1).any(axis=1).astype(int)
    # Last 7 rows of each SKU-store history have no complete future window.
    valid = g["Date"].transform(lambda s: s.shift(-7).notna())
    result = df.loc[valid, ["Date"] + FEATURES + ["stockout_within_7_days"]].copy()
    result.to_csv(save_path, index=False)
    return result

if __name__ == "__main__":
    out = build_dataset()
    print(f"Saved {len(out):,} rows to {PROCESSED}")
    print(out["stockout_within_7_days"].value_counts(normalize=True).rename("share"))
