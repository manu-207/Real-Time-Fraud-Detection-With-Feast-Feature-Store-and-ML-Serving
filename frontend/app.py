"""
Streamlit Frontend for Real-Time Fraud Detection
==================================================
A visual dashboard to interact with the fraud detection API.
"""

import streamlit as st
import requests
import numpy as np
import json
import os

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛡️ Real-Time Credit Card Fraud Detection")
st.markdown(
    "Powered by **XGBoost + Feast Feature Store + Redis** | "
    "Features served in <5ms from Redis online store"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API Status: {health['status']}")
        st.metric("Model Loaded", "✅" if health["model_loaded"] else "❌")
        st.metric("Feast Connected", "✅" if health["feast_connected"] else "❌")
        st.metric("Redis Connected", "✅" if health["redis_connected"] else "❌")
    except Exception as e:
        st.error(f"API Unreachable: {e}")

    st.divider()
    st.header("📊 Drift Monitoring")
    try:
        drift = requests.get(f"{API_URL}/drift-summary", timeout=5).json()
        st.metric("Drift Share", f"{drift['drift_share']:.2%}")
        st.metric("Amount Drift", "⚠️ Yes" if drift["amount_drift"] else "✅ No")
        st.metric("Prediction Drift", "⚠️ Yes" if drift["prediction_drift"] else "✅ No")
        st.metric("Predictions Buffered", drift["predictions_since_last_check"])
    except Exception:
        st.info("Drift data not available yet")

# ── Main Content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Transaction", "⚡ Feast Lookup", "📈 Batch Test"])

# ── Tab 1: Direct Prediction ─────────────────────────────────────────────────
with tab1:
    st.subheader("Submit a Transaction for Fraud Analysis")
    st.markdown("Enter transaction features to get a real-time fraud prediction.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Transaction Details**")
        transaction_id = st.number_input("Transaction ID", value=12345, step=1)
        amount = st.number_input("Amount ($)", value=150.0, min_value=0.0, step=10.0)
        hour = st.slider("Hour of Day", 0, 23, 14)
        is_night = 1 if (hour >= 23 or hour <= 5) else 0
        st.info(f"Night Transaction: {'Yes 🌙' if is_night else 'No ☀️'}")

    with col2:
        st.markdown("**Prediction Mode**")
        use_random = st.checkbox("Generate random PCA features (for testing)", value=True)

        if use_random:
            np.random.seed(transaction_id)
            v_features = np.random.randn(28)
            st.caption("Random V1-V28 features generated from transaction ID as seed")
        else:
            st.caption("Using default sample values")
            v_features = np.array([
                -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, -0.04,
                0.57, -0.38, -0.23, -0.04, -0.44, -0.12, -0.07, -0.23,
                -0.29, -0.11, 0.01, -0.15, -0.07, -0.23, -0.07, 0.56,
                -0.32, -0.09, -0.02, -0.05
            ])

    # Simulate fraud-like features
    simulate_fraud = st.checkbox("🚨 Simulate fraudulent transaction (shifts key features)")
    if simulate_fraud:
        v_features[0] -= 2.5   # V1
        v_features[2] -= 2.0   # V3
        v_features[13] -= 3.5  # V14
        v_features[16] -= 2.0  # V17

    if st.button("🔍 Predict Fraud", type="primary", use_container_width=True):
        # Build payload
        payload = {"transaction_id": int(transaction_id)}
        for i in range(28):
            payload[f"v{i+1}"] = round(float(v_features[i]), 4)
        payload["scaled_amount"] = round(float((amount - 88) / 88), 4)
        payload["scaled_time"] = round(float(np.random.randn()), 4)
        payload["hour_of_day"] = hour
        payload["is_night"] = is_night
        payload["amount_zscore"] = round(float((amount - 88) / 88), 4)

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()

                # Display result
                st.divider()
                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    if result["is_fraud"]:
                        st.error("🚨 FRAUD DETECTED")
                    else:
                        st.success("✅ LEGITIMATE")

                with r2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")

                with r3:
                    risk_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
                    st.metric("Risk Level", f"{risk_colors.get(result['risk_level'], '')} {result['risk_level'].upper()}")

                with r4:
                    st.metric("Latency", f"{result['latency_ms']:.1f} ms")

                # Show raw response
                with st.expander("View Raw API Response"):
                    st.json(result)
            else:
                st.error(f"API Error: {response.status_code} — {response.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")

# ── Tab 2: Feast Lookup ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Predict Using Feast Feature Store")
    st.markdown(
        "Only provide a **Transaction ID** — features are fetched from "
        "Redis online store in <5ms."
    )

    feast_id = st.number_input("Transaction ID (from materialized data)", value=100, step=1, key="feast_id")

    if st.button("⚡ Predict from Feast", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict/feast",
                json={"transaction_id": int(feast_id)},
                timeout=10,
            )
            if response.status_code == 200:
                result = response.json()

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if result["is_fraud"]:
                        st.error("🚨 FRAUD")
                    else:
                        st.success("✅ LEGIT")
                with c2:
                    st.metric("Probability", f"{result['fraud_probability']:.2%}")
                with c3:
                    st.metric("Risk", result["risk_level"].upper())
                with c4:
                    st.metric("Latency", f"{result['latency_ms']:.1f} ms")

                with st.expander("Raw Response"):
                    st.json(result)
            else:
                st.warning(f"Error {response.status_code}: {response.json().get('detail', response.text)}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# ── Tab 3: Batch Test ─────────────────────────────────────────────────────────
with tab3:
    st.subheader("Batch Prediction Test")
    st.markdown("Send multiple random transactions to test throughput and fraud detection rate.")

    n_transactions = st.slider("Number of transactions", 10, 200, 50)
    fraud_ratio = st.slider("Simulated fraud ratio", 0.0, 0.5, 0.1)

    if st.button("🚀 Run Batch Test", type="primary"):
        results = []
        progress = st.progress(0)

        for i in range(n_transactions):
            np.random.seed(i + 1000)
            is_simulated_fraud = np.random.random() < fraud_ratio

            v_feats = np.random.randn(28)
            if is_simulated_fraud:
                v_feats[0] -= 2.5
                v_feats[2] -= 2.0
                v_feats[13] -= 3.5
                v_feats[16] -= 2.0

            payload = {"transaction_id": i + 1000}
            for j in range(28):
                payload[f"v{j+1}"] = round(float(v_feats[j]), 4)
            payload["scaled_amount"] = round(float(np.random.randn()), 4)
            payload["scaled_time"] = round(float(np.random.randn()), 4)
            payload["hour_of_day"] = int(np.random.randint(0, 24))
            payload["is_night"] = 1 if payload["hour_of_day"] >= 23 or payload["hour_of_day"] <= 5 else 0
            payload["amount_zscore"] = round(float(np.random.randn()), 4)

            try:
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                if resp.status_code == 200:
                    results.append(resp.json())
            except Exception:
                pass

            progress.progress((i + 1) / n_transactions)

        if results:
            frauds = sum(1 for r in results if r["is_fraud"])
            avg_latency = np.mean([r["latency_ms"] for r in results])
            p95_latency = np.percentile([r["latency_ms"] for r in results], 95)

            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Predictions", len(results))
            with m2:
                st.metric("Frauds Detected", frauds)
            with m3:
                st.metric("Avg Latency", f"{avg_latency:.1f} ms")
            with m4:
                st.metric("P95 Latency", f"{p95_latency:.1f} ms")

            # Risk distribution
            risk_counts = {}
            for r in results:
                level = r["risk_level"]
                risk_counts[level] = risk_counts.get(level, 0) + 1
            st.bar_chart(risk_counts)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Streamlit | XGBoost + Feast + Redis + FastAPI | "
    "Real-time fraud detection MLOps pipeline"
)
