import sqlite3, json
from datetime import datetime
from pathlib import Path

DB_PATH = Path("fraud_app.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = get_conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS accounts (
        user_id INTEGER PRIMARY KEY,
        status TEXT DEFAULT 'active',
        locked_at TEXT, lock_reason TEXT,
        failed_logins INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE, user_id INTEGER, amount REAL,
        merchant_category TEXT, location TEXT, device_id TEXT,
        timestamp TEXT, risk_score REAL, decision TEXT, reasons TEXT
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, transaction_id TEXT, risk_score REAL,
        severity TEXT, message TEXT, created_at TEXT, resolved INTEGER DEFAULT 0
    );
    """)
    c.commit(); c.close()


def register_accounts(user_ids):
    c = get_conn()
    for uid in user_ids:
        c.execute("INSERT OR IGNORE INTO accounts (user_id) VALUES (?)", (int(uid),))
    c.commit(); c.close()


def lock_account(user_id, reason):
    c = get_conn()
    c.execute("UPDATE accounts SET status='locked', locked_at=?, lock_reason=? WHERE user_id=?",
              (datetime.now().isoformat(), reason, int(user_id)))
    c.commit(); c.close()


def unlock_account(user_id):
    c = get_conn()
    c.execute("UPDATE accounts SET status='active', locked_at=NULL, lock_reason=NULL, failed_logins=0 WHERE user_id=?",
              (int(user_id),))
    c.commit(); c.close()


def record_failed_login(user_id, max_attempts=5):
    c = get_conn()
    row = c.execute("SELECT failed_logins FROM accounts WHERE user_id=?", (int(user_id),)).fetchone()
    if not row:
        c.close(); return 0
    attempts = row["failed_logins"] + 1
    if attempts >= max_attempts:
        c.execute("UPDATE accounts SET failed_logins=?, status='locked', locked_at=?, lock_reason=? WHERE user_id=?",
                  (attempts, datetime.now().isoformat(), f"Too many failed sign-ins ({attempts})", int(user_id)))
    else:
        c.execute("UPDATE accounts SET failed_logins=? WHERE user_id=?", (attempts, int(user_id)))
    c.commit(); c.close()
    return attempts


def get_account(user_id):
    c = get_conn()
    row = c.execute("SELECT * FROM accounts WHERE user_id=?", (int(user_id),)).fetchone()
    c.close()
    return dict(row) if row else None


def all_accounts():
    c = get_conn()
    rows = c.execute("SELECT * FROM accounts ORDER BY user_id").fetchall()
    c.close()
    return [dict(r) for r in rows]


def record_transaction(tx, risk, decision, reasons):
    c = get_conn()
    c.execute("""INSERT OR REPLACE INTO transactions
        (transaction_id, user_id, amount, merchant_category, location, device_id,
         timestamp, risk_score, decision, reasons) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (tx["transaction_id"], int(tx["user_id"]), float(tx["amount"]),
         tx["merchant_category"], tx["location"], tx["device_id"],
         tx["timestamp"].isoformat() if hasattr(tx["timestamp"], "isoformat") else str(tx["timestamp"]),
         risk, decision, json.dumps(reasons)))
    c.commit(); c.close()


def add_alert(user_id, tx_id, risk, severity, message):
    c = get_conn()
    c.execute("""INSERT INTO alerts (user_id, transaction_id, risk_score, severity, message, created_at)
                 VALUES (?,?,?,?,?,?)""",
              (int(user_id), tx_id, risk, severity, message, datetime.now().isoformat()))
    c.commit(); c.close()


def get_alerts(limit=100):
    c = get_conn()
    rows = c.execute("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def get_transactions(limit=100):
    c = get_conn()
    rows = c.execute("SELECT * FROM transactions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]


def resolve_alert(alert_id):
    c = get_conn()
    c.execute("UPDATE alerts SET resolved=1 WHERE id=?", (int(alert_id),))
    c.commit(); c.close()


def reset_demo():
    c = get_conn()
    c.executescript("DELETE FROM transactions; DELETE FROM alerts; UPDATE accounts SET status='active', locked_at=NULL, lock_reason=NULL, failed_logins=0;")
    c.commit(); c.close()