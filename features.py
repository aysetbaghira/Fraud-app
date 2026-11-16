import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "amount", "hour", "day_of_week", "amount_ratio", "amount_zscore",
    "velocity_1h", "velocity_24h", "distinct_merchants_24h",
    "is_new_location", "is_new_device", "is_night",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    stats = df.groupby("user_id")["amount"].agg(["mean", "std"]).reset_index()
    stats.columns = ["user_id", "user_avg_amount", "user_std_amount"]
    stats["user_std_amount"] = stats["user_std_amount"].fillna(stats["user_avg_amount"] * 0.3)
    df = df.merge(stats, on="user_id", how="left")

    df["amount_ratio"] = df["amount"] / df["user_avg_amount"].clip(lower=1)
    df["amount_zscore"] = (df["amount"] - df["user_avg_amount"]) / df["user_std_amount"].clip(lower=1)

    v1h, v24h, dm24 = [], [], []
    for _, grp in df.groupby("user_id"):
        times = grp["timestamp"]
        cats = grp["merchant_category"].values
        for i in range(len(grp)):
            t = times.iloc[i]
            m1 = (times >= t - pd.Timedelta(hours=1)) & (times <= t)
            m24 = (times >= t - pd.Timedelta(hours=24)) & (times <= t)
            v1h.append(int(m1.sum()))
            v24h.append(int(m24.sum()))
            dm24.append(len(set(cats[m24.values])))
    df["velocity_1h"] = v1h
    df["velocity_24h"] = v24h
    df["distinct_merchants_24h"] = dm24

    df["is_new_location"] = 0
    df["is_new_device"] = 0
    for _, grp in df.groupby("user_id"):
        seen_l, seen_d = set(), set()
        for i in grp.index:
            loc, dev = df.at[i, "location"], df.at[i, "device_id"]
            df.at[i, "is_new_location"] = 0 if loc in seen_l else 1
            df.at[i, "is_new_device"] = 0 if dev in seen_d else 1
            seen_l.add(loc); seen_d.add(dev)

    df["is_night"] = ((df["hour"] < 6) | (df["hour"] > 23)).astype(int)
    return df


def features_for_new_tx(new_tx, user_history: pd.DataFrame, user_stats: dict) -> dict:
    history = user_history.sort_values("timestamp")
    t = new_tx["timestamp"]
    u_avg = user_stats["user_avg_amount"]
    u_std = max(user_stats["user_std_amount"], 1.0)

    times = history["timestamp"]
    m1 = (times >= t - pd.Timedelta(hours=1)) & (times < t)
    m24 = (times >= t - pd.Timedelta(hours=24)) & (times < t)

    merchants = set(history.loc[m24, "merchant_category"].tolist())
    merchants.add(new_tx["merchant_category"])

    return {
        "amount": float(new_tx["amount"]),
        "hour": t.hour,
        "day_of_week": t.dayofweek,
        "amount_ratio": new_tx["amount"] / max(u_avg, 1),
        "amount_zscore": (new_tx["amount"] - u_avg) / u_std,
        "velocity_1h": int(m1.sum()) + 1,
        "velocity_24h": int(m24.sum()) + 1,
        "distinct_merchants_24h": len(merchants),
        "is_new_location": int(new_tx["location"] not in set(history["location"])),
        "is_new_device": int(new_tx["device_id"] not in set(history["device_id"])),
        "is_night": int(t.hour < 6 or t.hour > 23),
    }