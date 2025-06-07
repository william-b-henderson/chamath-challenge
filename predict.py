import sys
from decision_tree import load_model, predict_new_case


def main():
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print(
            "Usage: python predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>"
        )
        sys.exit(1)

    try:
        # Parse command line arguments
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])

        # Load the saved model
        dt = load_model(verbose=False)
        if dt is None:
            print(
                "Error: No trained model found. Please run decision_tree.py first to train the model."
            )
            sys.exit(1)

        # Make prediction
        prediction = predict_new_case(
            dt, trip_duration_days, miles_traveled, total_receipts_amount
        )
        print(f"{prediction:.2f}")

    except ValueError as e:
        print(f"Error: Invalid input - {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
