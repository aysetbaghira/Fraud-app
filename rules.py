def evaluate_rules(tx: dict, user_stats: dict):
    u_avg = max(user_stats.get("user_avg_amount", 1), 1)
    ratio = tx["amount"] / u_avg

    rules = [
        ("AMOUNT_SPIKE",    ratio > 5,                0.25, f"Amount is {ratio:.1f}x user's average"),
        ("NEW_LOCATION",    tx.get("is_new_location", 0) == 1, 0.15, "Transaction from a new location"),
        ("NEW_DEVICE",      tx.get("is_new_device", 0) == 1,   0.20, "Transaction from an unknown device"),
        ("HIGH_VELOCITY",   tx.get("velocity_1h", 0) >= 5,     0.20, f"{tx.get('velocity_1h', 0)} transactions in last hour"),
        ("NIGHT_HIGH_VALUE", tx.get("is_night", 0) == 1 and ratio > 3, 0.15, "High-value transaction at night"),
        ("MANY_MERCHANTS",  tx.get("distinct_merchants_24h", 0) > 6, 0.10, "Many distinct merchants in 24h"),
    ]
    return rules


def rules_score(rules):
    total, triggered = 0.0, []
    for name, trig, weight, desc in rules:
        if trig:
            total += weight
            triggered.append({"rule": name, "weight": weight, "description": desc})
    return min(total, 1.0), triggered