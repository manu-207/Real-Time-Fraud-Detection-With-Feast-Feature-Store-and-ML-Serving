#!/bin/bash
# ── Load Test: Send random predictions to the API ─────────────────────────────
# Generates random transaction data and sends it to the /predict endpoint.
# Useful for testing drift detection and Prometheus metrics.

API_URL="${1:-http://localhost:8000/predict}"
NUM_REQUESTS="${2:-200}"

echo "═══════════════════════════════════════════════════════"
echo "  Sending $NUM_REQUESTS predictions to $API_URL"
echo "═══════════════════════════════════════════════════════"

for i in $(seq 1 $NUM_REQUESTS); do
    # Generate random PCA features
    V1=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V2=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V3=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V4=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V5=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V6=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V7=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V8=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V9=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V10=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V11=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V12=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V13=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V14=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V15=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V16=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V17=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V18=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V19=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V20=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V21=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V22=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V23=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V24=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V25=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V26=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V27=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)
    V28=$(echo "scale=4; ($RANDOM - 16384) / 8192" | bc)

    AMOUNT=$(echo "scale=2; $RANDOM / 100" | bc)
    HOUR=$((RANDOM % 24))
    IS_NIGHT=$( [ $HOUR -ge 23 ] || [ $HOUR -le 5 ] && echo 1 || echo 0 )

    RESPONSE=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"transaction_id\": $i,
            \"v1\": $V1, \"v2\": $V2, \"v3\": $V3, \"v4\": $V4,
            \"v5\": $V5, \"v6\": $V6, \"v7\": $V7, \"v8\": $V8,
            \"v9\": $V9, \"v10\": $V10, \"v11\": $V11, \"v12\": $V12,
            \"v13\": $V13, \"v14\": $V14, \"v15\": $V15, \"v16\": $V16,
            \"v17\": $V17, \"v18\": $V18, \"v19\": $V19, \"v20\": $V20,
            \"v21\": $V21, \"v22\": $V22, \"v23\": $V23, \"v24\": $V24,
            \"v25\": $V25, \"v26\": $V26, \"v27\": $V27, \"v28\": $V28,
            \"scaled_amount\": $AMOUNT, \"scaled_time\": -0.5,
            \"hour_of_day\": $HOUR, \"is_night\": $IS_NIGHT,
            \"amount_zscore\": 0.3
        }")

    FRAUD=$(echo $RESPONSE | python -c "import sys,json; print(json.load(sys.stdin).get('is_fraud','?'))" 2>/dev/null)
    echo "[$i/$NUM_REQUESTS] is_fraud=$FRAUD"

    sleep 0.2
done

echo ""
echo "✅ Done! Check /metrics and /drift-summary for results."
