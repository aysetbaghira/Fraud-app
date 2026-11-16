import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from core.data import generate_transactions, MERCHANT_CATEGORIES, LOCATIONS
from core.features import engineer_features, features_for_new_tx, FEATURE_COLUMNS
from core.models import train_models, score_transaction
from core.rules import evaluate_rules, rules_score
from core import db

st.set_page_config(page_title="FraudShield · Fraud Detection", page_icon="🛡️", layout="wide")

# ─── Theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #eef2f9 100%); }
    h1, h2, h3 { color: #1f2a44; }
    .brand { font-size: 2.4rem; font-weight: 800;
             background: linear-gradient(90deg,#4f46e5 0%,#7c3aed 50%,#db2777 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .card { background: white; border-radius: 14px; padding: 1.1rem 1.3rem;
            box-shadow: 0 4px 14px rgba(31,42,68,0.06); border: 1px solid #eef1f6; }
    .kpi { font-size: 1.9rem; font-weight: 700; color: #1f2a44; }
    .kpi-label { font-size: 0.8rem; color: #7a86a1; text-transform: uppercase; letter-spacing: 0.5px; }
    .risk-high { color: #e11d48; font-weight: 700; }
    .risk-mid  { color: #f59e0b; font-weight: 700; }
    .risk-low  { color: #10b981; font-weight: 700; }
    .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .pill-high { background: #fee2e2; color: #b91c1c; }
    .pill-mid  { background: #fef3c7; color: #92400e; }
    .pill-low  { background: #d1fae5; color: #065f46; }
</style>
""", unsafe_allow_html=True)


# ─── Bootstrap ─────────────────────────────────────────────────────────
@st.cache_resource
def bootstrap():
    db.init_db()
    raw = generate_transactions()
    feat = engineer_features(raw)
    models = train_models(feat)
    db.register_accounts(feat["user_id"].unique().tolist())
    return feat, models

df_feat, models = bootstrap()
user_stats_map = (
    df_feat.groupby("user_id")[["user_avg_amount", "user_std_amount"]]
    .first().reset_index().set_index("user_id").to_dict("index")
)


def get_user_history(uid):
    return df_feat[df_feat["user_id"] == uid].copy()


def score_and_store(user_id, amount, category, location, device, simulate_failed_logins=0):
    user_id = int(user_id)
    account = db.get_account(user_id) or {"status": "active"}

    if account["status"] == "locked":
        return {"blocked": True, "reason": "Account is locked", "risk": 1.0, "rules": [], "scores": {}}

    # Simulate failed logins if requested
    for _ in range(simulate_failed_logins):
        db.record_failed_login(user_id)
    account = db.get_account(user_id) or {"status": "active"}

    history = get_user_history(user_id)
    stats = user_stats_map[user_id]
    new_tx = {
        "transaction_id": f"TX{np.random.randint(10_000_000, 99_999_999)}",
        "user_id": user_id, "amount": float(amount), "merchant_category": category,
        "location": location, "device_id": device, "timestamp": datetime.now(),
    }
    feats = features_for_new_tx(new_tx, history, stats)
    ml_risk, score_breakdown = score_transaction(feats, models)
    rule_list = evaluate_rules({**new_tx, **feats}, stats)
    rule_risk, triggered = rules_score(rule_list)

    final_risk = float(np.clip(0.55 * ml_risk + 0.45 * rule_risk, 0, 1))

    if final_risk >= 0.75:
        decision = "BLOCK"
        db.lock_account(user_id, f"Auto-locked: risk {final_risk:.2f}")
        db.add_alert(user_id, new_tx["transaction_id"], final_risk, "HIGH",
                     f"Auto-locked. Reason: {', '.join([t['rule'] for t in triggered]) or 'high ML score'}")
    elif final_risk >= 0.45:
        decision = "REVIEW"
        db.add_alert(user_id, new_tx["transaction_id"], final_risk, "MEDIUM",
                     f"Flagged for review. {len(triggered)} rule(s) triggered.")
    else:
        decision = "APPROVE"

    db.record_transaction(new_tx, final_risk, decision, triggered)

    return {
        "blocked": decision == "BLOCK",
        "decision": decision,
        "risk": final_risk,
        "ml_risk": ml_risk,
        "rule_risk": rule_risk,
        "scores": score_breakdown,
        "rules": triggered,
        "features": feats,
        "tx": new_tx,
    }


# ─── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🛡️ FraudShield</div>', unsafe_allow_html=True)
    st.caption("Real-time transaction fraud detection")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard", "💳 Score a Transaction", "👤 User Profiles",
        "🔔 Alerts & Accounts", "📈 Model Analytics"
    ])
    st.markdown("---")
    if st.button("♻️ Reset demo data", use_container_width=True):
        db.reset_demo(); st.cache_resource.clear(); st.rerun()


# ═══════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("## 📊 Operations Dashboard")

    all_tx = db.get_transactions(1000)
    alerts = db.get_alerts(1000)
    accounts = db.all_accounts()
    fraud_alerts = [a for a in alerts if a["severity"] == "HIGH"]
    locked = [a for a in accounts if a["status"] == "locked"]
    blocked_tx = [t for t in all_tx if t["decision"] == "BLOCK"]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, color in [
        (c1, "Transactions Scored", len(all_tx), "#4f46e5"),
        (c2, "High-Risk Alerts", len(fraud_alerts), "#e11d48"),
        (c3, "Accounts Locked", len(locked), "#f59e0b"),
        (c4, "Blocked Transactions", len(blocked_tx), "#7c3aed"),
    ]:
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="kpi-label">{label}</div>
                <div class="kpi" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###")
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown("#### Recent activity")
        if all_tx:
            recent = pd.DataFrame(all_tx[:15])[["transaction_id", "user_id", "amount", "decision", "risk_score", "timestamp"]]
            recent["risk_score"] = recent["risk_score"].round(2)
            recent["amount"] = recent["amount"].map(lambda x: f"₹{x:,.0f}")
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions yet. Score one from the **💳 Score a Transaction** page.")

    with col_b:
        st.markdown("#### Decisions breakdown")
        if all_tx:
            counts = pd.Series([t["decision"] for t in all_tx]).value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.55,
                marker=dict(colors=["#10b981", "#f59e0b", "#e11d48"])
            ))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=260, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet.")

    st.markdown("#### Top users by flagged transactions")
    if all_tx:
        flagged = [t for t in all_tx if t["decision"] != "APPROVE"]
        if flagged:
            tmp = pd.DataFrame(flagged)
            top = tmp.groupby("user_id").size().reset_index(name="flagged").sort_values("flagged", ascending=False).head(10)
            fig = px.bar(top, x="user_id", y="flagged", color="flagged",
                         color_continuous_scale=["#fde68a", "#e11d48"])
            fig.update_layout(height=300, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: SCORE A TRANSACTION
# ═══════════════════════════════════════════════════════════════════════
elif page == "💳 Score a Transaction":
    st.markdown("## 💳 Transaction Simulator")
    st.caption("Enter a transaction. The model scores it live against the user's history.")

    user_ids = sorted(user_stats_map.keys())
    with st.form("score_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        uid = c1.selectbox("User", user_ids, format_func=lambda u: f"User #{u}")
        amount = c1.number_input("Amount (₹)", min_value=1.0, value=5000.0, step=100.0)

        category = c2.selectbox("Merchant category", MERCHANT_CATEGORIES)
        location = c2.selectbox("Location", LOCATIONS)

        device = c3.text_input("Device ID", value="device_unknown_01")
        failed = c3.slider("Simulate failed sign-ins", 0, 5, 0)

        submitted = st.form_submit_button("🔍 Score Transaction", use_container_width=True)

    if submitted:
        result = score_and_store(uid, amount, category, location, device, failed)
        st.markdown("###")

        if result.get("blocked") and result.get("decision") is None:
            st.error(f"🚫 **{result['reason']}** — please unlock from **Alerts & Accounts**.")
        else:
            cA, cB = st.columns([1, 2])

            with cA:
                color = "#10b981" if result["risk"] < 0.45 else ("#f59e0b" if result["risk"] < 0.75 else "#e11d48")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=result["risk"] * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "Fraud Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 45], "color": "#d1fae5"},
                            {"range": [45, 75], "color": "#fef3c7"},
                            {"range": [75, 100], "color": "#fee2e2"},
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(t=20, b=10, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

            with cB:
                dec = result["decision"]
                cls = {"APPROVE": "risk-low", "REVIEW": "risk-mid", "BLOCK": "risk-high"}.get(dec, "")
                st.markdown(f"""
                <div class="card">
                    <div class="kpi-label">Decision</div>
                    <div class="kpi {cls}" style="font-size:2.2rem">{dec}</div>
                    <hr style="margin:0.6rem 0">
                    <div style="display:flex; gap:2rem">
                        <div><div class="kpi-label">ML Score</div><div style="font-weight:700">{result['ml_risk']:.2f}</div></div>
                        <div><div class="kpi-label">Rules Score</div><div style="font-weight:700">{result['rule_risk']:.2f}</div></div>
                        <div><div class="kpi-label">Final</div><div style="font-weight:700">{result['risk']:.2f}</div></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("#### 🚩 Triggered rules")
            if result["rules"]:
                for r in result["rules"]:
                    sev = "pill-high" if r["weight"] >= 0.2 else "pill-mid"
                    st.markdown(
                        f'<span class="pill {sev}">{r["rule"]}</span> &nbsp; {r["description"]} '
                        f'<span style="color:#94a3b8">(+{r["weight"]:.2f})</span>',
                        unsafe_allow_html=True
                    )
            else:
                st.success("✅ No rules triggered — transaction looks consistent with the user's profile.")

            with st.expander("🔎 Feature snapshot"):
                st.json(result["features"])


# ═══════════════════════════════════════════════════════════════════════
# PAGE: USER PROFILES
# ═══════════════════════════════════════════════════════════════════════
elif page == "👤 User Profiles":
    st.markdown("## 👤 User Behavioral Profiles")
    user_ids = sorted(user_stats_map.keys())
    uid = st.selectbox("Select a user", user_ids, format_func=lambda u: f"User #{u}")
    hist = get_user_history(uid).sort_values("timestamp")
    acct = db.get_account(uid)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Spend", f"₹{hist['amount'].mean():,.0f}")
    c2.metric("Max Spend", f"₹{hist['amount'].max():,.0f}")
    c3.metric("Transactions", f"{len(hist)}")
    status = acct["status"] if acct else "active"
    c4.metric("Account Status", "🔒 Locked" if status == "locked" else "✅ Active")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Spending over time")
        fig = px.scatter(hist, x="timestamp", y="amount",
                         color=hist["is_fraud"].map({0: "Normal", 1: "Fraud (label)"}),
                         color_discrete_map={"Normal": "#4f46e5", "Fraud (label)": "#e11d48"},
                         labels={"amount": "Amount (₹)"}, hover_data=["merchant_category", "location"])
        fig.update_layout(height=320, legend_title="", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Category breakdown")
        cat = hist.groupby("merchant_category")["amount"].sum().sort_values(ascending=False)
        fig = px.bar(x=cat.values, y=cat.index, orientation="h",
                     color=cat.values, color_continuous_scale="Purples")
        fig.update_layout(height=320, showlegend=False, xaxis_title="Total ₹", yaxis_title="",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Location history")
    loc = hist.groupby("location").agg(txns=("transaction_id", "count"), total=("amount", "sum")).reset_index()
    st.dataframe(loc, use_container_width=True, hide_index=True)

    st.markdown("#### Recent transactions")
    recent = hist.tail(15)[["transaction_id", "timestamp", "amount", "merchant_category", "location", "device_id"]].iloc[::-1]
    st.dataframe(recent, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: ALERTS & ACCOUNTS
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔔 Alerts & Accounts":
    st.markdown("## 🔔 Alerts & Account Management")

    tab1, tab2 = st.tabs(["🚨 Fraud Alerts", "🔐 Account Locks"])

    with tab1:
        alerts = db.get_alerts(200)
        if not alerts:
            st.info("No alerts yet. Score some risky transactions to generate alerts.")
        else:
            for a in alerts:
                sev = a["severity"]
                cls = {"HIGH": "pill-high", "MEDIUM": "pill-mid", "LOW": "pill-low"}.get(sev, "pill-mid")
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(
                        f'<div class="card" style="margin-bottom:0.4rem">'
                        f'<span class="pill {cls}">{sev}</span>&nbsp;'
                        f'<b>User #{a["user_id"]}</b> · {a["transaction_id"]} · '
                        f'risk <b>{a["risk_score"]:.2f}</b><br>'
                        f'<span style="color:#64748b; font-size:0.9rem">{a["message"]}</span><br>'
                        f'<span style="color:#94a3b8; font-size:0.75rem">{a["created_at"][:19].replace("T"," ")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    if not a["resolved"] and st.button("Resolve", key=f"res_{a['id']}"):
                        db.resolve_alert(a["id"]); st.rerun()
                    elif a["resolved"]:
                        st.caption("✅ resolved")

    with tab2:
        accounts = db.all_accounts()
        locked = [a for a in accounts if a["status"] == "locked"]
        if not locked:
            st.success("No accounts are currently locked.")
        else:
            for a in locked:
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown(
                        f'<div class="card" style="margin-bottom:0.4rem">'
                        f'🔒 <b>User #{a["user_id"]}</b><br>'
                        f'<span style="color:#64748b; font-size:0.9rem">{a["lock_reason"]}</span><br>'
                        f'<span style="color:#94a3b8; font-size:0.75rem">Locked at {a["locked_at"][:19].replace("T"," ")}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                with c2:
                    if st.button("Unlock", key=f"unlock_{a['user_id']}"):
                        db.unlock_account(a["user_id"]); st.rerun()

        st.markdown("#### All accounts")
        df_acc = pd.DataFrame(accounts)
        st.dataframe(df_acc, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📈 Model Analytics":
    st.markdown("## 📈 Model Analytics")

    # Evaluate on training data (in-sample; replace with held-out for real use)
    X = df_feat[FEATURE_COLUMNS].values
    y = df_feat["is_fraud"].values
    Xs = models["scaler"].transform(X)
    iso_pred = (models["iso"].predict(Xs) == -1).astype(int)
    clf_proba = models["clf"].predict_proba(X)[:, 1]
    combined = 0.6 * clf_proba + 0.4 * ((iso_pred > 0).astype(int))
    pred = (combined >= 0.5).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1 Score", f"{f1:.3f}")
    c4.metric("Fraud / Total", f"{y.sum()} / {len(y)}")

    st.markdown("###")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Confusion matrix")
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: Normal", "Pred: Fraud"], y=["True: Normal", "True: Fraud"])
        fig.update_layout(height=320, coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Score distribution")
        score_df = pd.DataFrame({"score": clf_proba, "label": y})
        fig = px.histogram(score_df, x="score", color="label", nbins=40,
                           color_discrete_map={0: "#4f46e5", 1: "#e11d48"},
                           labels={"label": "is_fraud"})
        fig.update_layout(height=320, bargap=0.02, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Top features (from Gradient Boosting)")
    imp = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": models["clf"].feature_importances_
    }).sort_values("importance", ascending=True)
    fig = px.bar(imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Purples")
    fig.update_layout(height=400, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ Metrics are computed in-sample on the synthetic dataset. "
               "For a production system, evaluate on a held-out test set.")