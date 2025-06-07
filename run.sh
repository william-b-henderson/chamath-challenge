#!/bin/bash

# Black Box Challenge - Decision Tree Implementation
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed!" >&2
    exit 1
fi

# Check if required files exist
if [ ! -f "predict.py" ] || [ ! -f "decision_tree.py" ] || [ ! -f "public_cases.json" ]; then
    echo "Error: Required files (predict.py, decision_tree.py, public_cases.json) not found!" >&2
    exit 1
fi

# Run the prediction script
python3 predict.py "$1" "$2" "$3"