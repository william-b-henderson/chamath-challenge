#!/usr/bin/env python3

import json
import sys
import os
from decision_tree import load_data, train_decision_tree, predict_new_case


def main():
    print("🧾 Black Box Challenge - Generating Private Results")
    print("====================================================")
    print()

    # Check if private cases exist
    if not os.path.exists("private_cases.json"):
        print("❌ Error: private_cases.json not found!")
        print("Please ensure the private cases file is in the current directory.")
        sys.exit(1)

    print("📊 Processing test cases and generating results...")
    print("📝 Output will be saved to private_results.txt")
    print()

    # Load and train the model using public cases
    print("Training decision tree model...")
    df = load_data("public_cases.json")
    dt, _, _, _, _, _ = train_decision_tree(df)

    # Load private test cases
    print("Loading private test cases...")
    with open("private_cases.json", "r") as f:
        private_cases = json.load(f)

    total_cases = len(private_cases)

    # Remove existing results file if it exists
    if os.path.exists("private_results.txt"):
        os.remove("private_results.txt")

    print(f"Processing {total_cases} test cases...")

    # Process each test case
    for i, case in enumerate(private_cases):
        if (i % 100) == 0 and i > 0:
            print(f"Progress: {i}/{total_cases} cases processed...")

        try:
            # Extract test case data
            trip_duration = case["trip_duration_days"]
            miles_traveled = case["miles_traveled"]
            receipts_amount = case["total_receipts_amount"]

            # Get prediction from decision tree
            prediction = predict_new_case(
                dt, trip_duration, miles_traveled, receipts_amount
            )

            # Write result to file
            with open("private_results.txt", "a") as f:
                f.write(f"{prediction}\n")

        except Exception as e:
            print(f"Error on case {i+1}: {str(e)}")
            with open("private_results.txt", "a") as f:
                f.write("ERROR\n")

    print()
    print("✅ Results generated successfully!")
    print("📄 Output saved to private_results.txt")
    print(
        "📊 Each line contains the result for the corresponding test case in private_cases.json"
    )
    print()
    print("🎯 Next steps:")
    print("  1. Check private_results.txt - it should contain one result per line")
    print(
        "  2. Each line corresponds to the same-numbered test case in private_cases.json"
    )
    print("  3. Lines with 'ERROR' indicate cases where prediction failed")
    print("  4. Submit your private_results.txt file when ready!")
    print()
    print("📈 File format:")
    print("  Line 1: Result for private_cases.json[0]")
    print("  Line 2: Result for private_cases.json[1]")
    print("  Line 3: Result for private_cases.json[2]")
    print("  ...")
    print("  Line N: Result for private_cases.json[N-1]")


if __name__ == "__main__":
    main()
