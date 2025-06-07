#!/bin/bash
# Usage: ./run.sh <trip_days> <miles> <receipts>

trip_days=$1
miles=$2
receipts=$3

echo "$trip_days $miles $receipts" | python3 solution/predict.py
