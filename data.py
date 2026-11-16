import numpy as np
import pandas as pd
from datetime import datetime, timedelta

MERCHANT_CATEGORIES = ["groceries", "restaurants", "electronics", "travel",
                       "fuel", "online_retail", "utilities", "entertainment",
                       "healthcare", "education"]
LOCATIONS = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata",
             "Hyderabad", "Pune", "Ahmedabad"]


def generate_transactions(n_users=50, n_days=30, fraud_rate=0.02, seed=42):
    rng = np.random.default_rng(seed)

    users = {}
    for uid in range(1, n_users + 1):
        users[uid] = {
            "avg_amount": rng.uniform(200, 5000),
            "std_amount": rng.uniform(50, 500),
            "home_location": str(rng.choice(LOCATIONS)),
            "preferred_categories": rng.choice(MERCHANT_CATEGORIES, size=3, replace=False).tolist(),
            "devices": [f"device_{uid}_{i}" for i in range(rng.integers(1, 3))],
            "hour_mean": rng.uniform(8, 22),
            "hour_std": rng.uniform(2, 4),
        }

    start = datetime.now() - timedelta(days=n_days)
    rows = []
    tx_id = 1

    for uid, prof in users.items():
        n_tx = int(rng.integers(20, 80))
        for _ in range(n_tx):
            day_off = int(rng.integers(0, n_days))
            hour = int(np.clip(rng.normal(prof["hour_mean"], prof["hour_std"]), 0, 23))
            ts = start + timedelta(days=day_off, hours=hour, minutes=int(rng.integers(0, 60)))
            amount = max(10, float(rng.normal(prof["avg_amount"], prof["std_amount"])))
            category = str(rng.choice(prof["preferred_categories"]))
            location = prof["home_location"] if rng.random() < 0.9 else str(rng.choice(LOCATIONS))
            device = str(rng.choice(prof["devices"]))
            is_fraud = 0

            if rng.random() < fraud_rate:
                kind = rng.choice(["amount_spike", "new_location", "velocity", "new_device"])
                if kind == "amount_spike":
                    amount = prof["avg_amount"] * rng.uniform(5, 15)
                elif kind == "new_location":
                    others = [l for l in LOCATIONS if l != prof["home_location"]]
                    location = str(rng.choice(others))
                    amount = prof["avg_amount"] * rng.uniform(2, 5)
                elif kind == "new_device":
                    device = f"unknown_{rng.integers(1000, 9999)}"
                    amount = prof["avg_amount"] * rng.uniform(2, 6)
                else:
                    amount = prof["avg_amount"] * rng.uniform(1, 3)
                is_fraud = 1

            rows.append({
                "transaction_id": f"TX{tx_id:07d}",
                "user_id": uid,
                "timestamp": ts,
                "amount": round(float(amount), 2),
                "merchant_category": category,
                "location": location,
                "device_id": device,
                "is_fraud": is_fraud,
            })
            tx_id += 1

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)