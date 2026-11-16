# 🛡️ FraudShield — Real-Time Credit Card Fraud Detection

A ready-to-run fraud detection app that **scores transactions live**, tracks
**user spending patterns**, **auto-locks accounts** on high-risk activity, and
**alerts** on suspicious behavior.

## Features
- 🧠 **Hybrid ML** — Isolation Forest (unsupervised) + Gradient Boosting (supervised)
- 📜 **Rules engine** — amount spikes, new locations/devices, velocity, night transactions
- 👤 **Per-user profiling** — spending averages, deviation scores, merchant diversity
- 🔒 **Auto-lock** — triggered by high risk or repeated failed sign-ins
- 🔔 **Alerts & resolution workflow**
- 📊 **Analytics** — confusion matrix, score distribution, feature importance
- 💾 **SQLite persistence** — survives restarts

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py