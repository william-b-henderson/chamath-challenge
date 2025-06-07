import sys
import pickle
import numpy as np
from decision_tree import predict_new_case, load_data, train_decision_tree


def main():
    # Check if we have the required arguments
    if len(sys.argv) != 4:
        print(
            "Usage: python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>"
        )
        sys.exit(1)

    try:
        # Parse arguments
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])

        # Load and train the model
        df = load_data()
        dt, _, _, _, _, _ = train_decision_tree(df, max_depth=10)

        # Make prediction
        prediction = predict_new_case(
            dt, trip_duration_days, miles_traveled, total_receipts_amount
        )

        # Print just the prediction number (required by eval.sh)
        print(f"{prediction:.2f}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
