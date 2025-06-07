import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import seaborn as sns


def engineer_features(df):
    """Add engineered features based on interview insights"""
    # Efficiency features (from Kevin's insights)
    df["miles_per_day"] = df["miles_traveled"] / df["trip_duration_days"]
    df["is_efficient_trip"] = (df["miles_per_day"] >= 180) & (
        df["miles_per_day"] <= 220
    )
    df["is_very_efficient_trip"] = (df["miles_per_day"] > 220) & (
        df["miles_per_day"] <= 300
    )
    df["is_over_efficient_trip"] = (
        df["miles_per_day"] > 300
    )  # Kevin mentioned penalties for too high efficiency

    # Spending patterns (from Kevin and Lisa's insights)
    df["daily_spending"] = df["total_receipts_amount"] / df["trip_duration_days"]
    df["is_optimal_spending"] = (
        ((df["trip_duration_days"] <= 3) & (df["daily_spending"] <= 75))
        | (
            (df["trip_duration_days"] > 3)
            & (df["trip_duration_days"] <= 6)
            & (df["daily_spending"] <= 120)
        )
        | ((df["trip_duration_days"] > 6) & (df["daily_spending"] <= 90))
    )
    df["is_high_spending"] = df["daily_spending"] > 120
    df["is_low_spending"] = df["daily_spending"] < 50
    df["spending_to_miles_ratio"] = (
        df["daily_spending"] / df["miles_per_day"]
    )  # Interaction effect mentioned by Kevin

    # Trip length features (from Jennifer and Lisa's insights)
    df["is_sweet_spot_trip"] = (df["trip_duration_days"] >= 4) & (
        df["trip_duration_days"] <= 6
    )
    df["is_five_day_trip"] = (
        df["trip_duration_days"] == 5
    )  # Kevin mentioned 5-day trips get bonuses
    df["is_long_trip"] = df["trip_duration_days"] > 8  # Kevin's "vacation penalty"
    df["is_short_trip"] = df["trip_duration_days"] <= 3

    # Mileage tiers (from Lisa's insights)
    df["mileage_tier"] = pd.cut(
        df["miles_traveled"],
        bins=[0, 100, 300, 600, float("inf")],
        labels=["tier1", "tier2", "tier3", "tier4"],
    )
    df["mileage_tier"] = df["mileage_tier"].cat.codes

    # Receipt amount features (from Lisa and Dave's insights)
    df["receipts_per_day"] = df["total_receipts_amount"] / df["trip_duration_days"]
    df["is_low_receipts"] = df["receipts_per_day"] < 50
    df["is_high_receipts"] = df["receipts_per_day"] > 200
    df["is_medium_receipts"] = (df["receipts_per_day"] >= 50) & (
        df["receipts_per_day"] <= 200
    )
    df["receipts_ends_in_49_or_99"] = df["total_receipts_amount"].apply(
        lambda x: str(x).split(".")[-1] in ["49", "99"] if "." in str(x) else False
    )  # Lisa's rounding observation

    # Trip type categorization (from Kevin's insights)
    df["trip_type"] = pd.cut(
        df["miles_per_day"],
        bins=[0, 100, 180, 220, 300, float("inf")],
        labels=[
            "very_low_efficiency",
            "low_efficiency",
            "medium_efficiency",
            "high_efficiency",
            "very_high_efficiency",
        ],
    )
    df["trip_type"] = df["trip_type"].cat.codes

    # Interaction features (from Kevin's insights about factor interactions)
    df["efficiency_spending_score"] = df["miles_per_day"] * (
        1 / df["daily_spending"]
    )  # Higher is better
    df["trip_length_efficiency"] = df["trip_duration_days"] * df["miles_per_day"]
    df["is_sweet_spot_combo"] = (
        (df["trip_duration_days"] == 5)
        & (df["miles_per_day"] >= 180)
        & (df["daily_spending"] <= 100)
    )  # Kevin's "sweet spot combo"
    df["is_vacation_penalty"] = (df["trip_duration_days"] >= 8) & (
        df["daily_spending"] > 100
    )  # Kevin's "vacation penalty"

    # Spending efficiency features (from Lisa's insights about diminishing returns)
    df["spending_efficiency"] = df["total_receipts_amount"] / df["miles_traveled"]
    df["is_optimal_spending_range"] = (df["total_receipts_amount"] >= 600) & (
        df["total_receipts_amount"] <= 800
    )  # Lisa's observation about medium-high amounts

    # Trip characteristics (from Marcus and Jennifer's insights)
    df["is_balanced_trip"] = (
        (df["miles_per_day"] >= 100)
        & (df["miles_per_day"] <= 300)
        & (df["daily_spending"] >= 50)
        & (df["daily_spending"] <= 150)
    )
    df["is_high_mileage_low_spending"] = (df["miles_per_day"] > 200) & (
        df["daily_spending"] < 75
    )
    df["is_low_mileage_high_spending"] = (df["miles_per_day"] < 100) & (
        df["daily_spending"] > 150
    )

    return df


def load_data(filename="public_cases.json"):
    """Load the JSON data and convert to pandas DataFrame with engineered features"""
    with open(filename, "r") as f:
        data = json.load(f)

    # Extract features and target
    features = []
    targets = []

    for case in data:
        features.append(
            [
                case["input"]["trip_duration_days"],
                case["input"]["miles_traveled"],
                case["input"]["total_receipts_amount"],
            ]
        )
        targets.append(case["expected_output"])

    # Create DataFrame
    df = pd.DataFrame(
        features,
        columns=["trip_duration_days", "miles_traveled", "total_receipts_amount"],
    )
    df["expected_output"] = targets

    # Add engineered features
    df = engineer_features(df)

    return df


def train_decision_tree(
    df,
    max_depth=10,  # Increased depth to capture more complex patterns
    min_samples_split=5,  # Increased to prevent overfitting
    min_samples_leaf=3,  # Increased to prevent overfitting
    random_state=42,
):
    """Train a decision tree regressor with engineered features"""

    # Select features for training
    feature_columns = [
        # Original features
        "trip_duration_days",
        "miles_traveled",
        "total_receipts_amount",
        # Engineered features
        "miles_per_day",
        "is_efficient_trip",
        "daily_spending",
        "is_optimal_spending",
        "is_sweet_spot_trip",
        "is_five_day_trip",
        "mileage_tier",
        "receipts_per_day",
        "is_low_receipts",
        "is_high_receipts",
        "trip_type",
    ]

    X = df[feature_columns]
    y = df["expected_output"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create and train the decision tree with increased complexity
    dt = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    dt.fit(X_train, y_train)

    # Make predictions
    y_train_pred = dt.predict(X_train)
    y_test_pred = dt.predict(X_test)

    # Calculate metrics
    # train_mse = mean_squared_error(y_train, y_train_pred)
    # test_mse = mean_squared_error(y_test, y_test_pred)
    # train_r2 = r2_score(y_train, y_train_pred)
    # test_r2 = r2_score(y_test, y_test_pred)
    # train_mae = mean_absolute_error(y_train, y_train_pred)
    # test_mae = mean_absolute_error(y_test, y_test_pred)

    # print("Decision Tree Results:")
    # print(f"Training MSE: {train_mse:.2f}")
    # print(f"Testing MSE: {test_mse:.2f}")
    # print(f"Training R²: {train_r2:.4f}")
    # print(f"Testing R²: {test_r2:.4f}")
    # print(f"Training MAE: {train_mae:.2f}")
    # print(f"Testing MAE: {test_mae:.2f}")

    return dt, X_train, X_test, y_train, y_test, y_test_pred


def analyze_feature_importance(dt, feature_names=None):
    """Analyze and display feature importance"""
    if feature_names is None:
        feature_names = [
            "trip_duration_days",
            "miles_traveled",
            "total_receipts_amount",
            "miles_per_day",
            "is_efficient_trip",
            "daily_spending",
            "is_optimal_spending",
            "is_sweet_spot_trip",
            "is_five_day_trip",
            "mileage_tier",
            "receipts_per_day",
            "is_low_receipts",
            "is_high_receipts",
            "trip_type",
        ]

    importance = dt.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="importance", y="feature")
    plt.title("Feature Importance in Decision Tree")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

    return feature_importance


def display_tree_rules(dt, feature_names=None, max_depth=3):
    """Display the decision tree rules"""
    if feature_names is None:
        feature_names = [
            "trip_duration_days",
            "miles_traveled",
            "total_receipts_amount",
            "miles_per_day",
            "is_efficient_trip",
            "daily_spending",
            "is_optimal_spending",
            "is_sweet_spot_trip",
            "is_five_day_trip",
            "mileage_tier",
            "receipts_per_day",
            "is_low_receipts",
            "is_high_receipts",
            "trip_type",
        ]
    tree_rules = export_text(dt, feature_names=feature_names, max_depth=max_depth)
    print(f"\nDecision Tree Rules (showing first {max_depth} levels):")
    print(tree_rules)


def predict_new_case(dt, trip_duration_days, miles_traveled, total_receipts_amount):
    """Predict expected output for a new case with engineered features"""
    # Create a single-row DataFrame with the input features
    input_df = pd.DataFrame(
        [[trip_duration_days, miles_traveled, total_receipts_amount]],
        columns=["trip_duration_days", "miles_traveled", "total_receipts_amount"],
    )

    # Add engineered features
    input_df = engineer_features(input_df)

    # Select the same features used in training
    feature_columns = [
        "trip_duration_days",
        "miles_traveled",
        "total_receipts_amount",
        "miles_per_day",
        "is_efficient_trip",
        "daily_spending",
        "is_optimal_spending",
        "is_sweet_spot_trip",
        "is_five_day_trip",
        "mileage_tier",
        "receipts_per_day",
        "is_low_receipts",
        "is_high_receipts",
        "trip_type",
    ]

    # Make prediction
    prediction = dt.predict(input_df[feature_columns])
    return prediction[0]


def plot_predictions_vs_actual(y_test, y_test_pred):
    """Plot predicted vs actual values"""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.tight_layout()
    plt.show()


def explore_data(df):
    """Explore the dataset"""
    print("Dataset Overview:")
    print(f"Shape: {df.shape}")
    print(f"\nDescriptive Statistics:")
    print(df.describe())

    print(f"\nCorrelation Matrix:")
    correlation_matrix = df.corr()
    print(correlation_matrix)

    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()


def optimize_tree_depth(df):
    """Find optimal tree depth using cross-validation"""
    X = df[["trip_duration_days", "miles_traveled", "total_receipts_amount"]]
    y = df["expected_output"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    depths = range(1, 21)
    train_scores = []
    test_scores = []

    for depth in depths:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)

        train_pred = dt.predict(X_train)
        test_pred = dt.predict(X_test)

        train_scores.append(r2_score(y_train, train_pred))
        test_scores.append(r2_score(y_test, test_pred))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(depths, train_scores, "o-", label="Training R²", color="blue")
    plt.plot(depths, test_scores, "o-", label="Testing R²", color="red")
    plt.xlabel("Tree Depth")
    plt.ylabel("R² Score")
    plt.title("Model Performance vs Tree Depth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    optimal_depth = depths[np.argmax(test_scores)]
    print(f"Optimal tree depth: {optimal_depth}")
    print(f"Best test R² score: {max(test_scores):.4f}")

    return optimal_depth


def evaluate_model(dt, df):
    """Evaluate the model's performance on the test cases using the same metrics as eval_decision_tree.sh"""
    # Get all test cases
    test_cases = df[
        [
            "trip_duration_days",
            "miles_traveled",
            "total_receipts_amount",
            "expected_output",
        ]
    ].values

    # Initialize counters
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0
    max_error = 0
    max_error_case = None
    results = []

    # Process each test case
    for i, (trip_duration, miles_traveled, receipts_amount, expected) in enumerate(
        test_cases
    ):
        # Get prediction
        prediction = predict_new_case(
            dt, trip_duration, miles_traveled, receipts_amount
        )

        # Calculate absolute error
        error = abs(prediction - expected)

        # Store result
        results.append(
            {
                "case_num": i + 1,
                "expected": expected,
                "actual": prediction,
                "error": error,
                "trip_duration": trip_duration,
                "miles_traveled": miles_traveled,
                "receipts_amount": receipts_amount,
            }
        )

        successful_runs += 1

        # Check for exact match (within $0.01)
        if error < 0.01:
            exact_matches += 1

        # Check for close match (within $1.00)
        if error < 1.0:
            close_matches += 1

        # Update total error
        total_error += error

        # Track maximum error
        if error > max_error:
            max_error = error
            max_error_case = f"Case {i+1}: {trip_duration} days, {miles_traveled} miles, ${receipts_amount:.2f} receipts"

    # Calculate metrics
    num_cases = len(test_cases)
    avg_error = total_error / successful_runs
    exact_pct = (exact_matches * 100) / successful_runs
    close_pct = (close_matches * 100) / successful_runs

    # Calculate score (same formula as eval_decision_tree.sh)
    score = avg_error * 100 + (num_cases - exact_matches) * 0.1

    # Print results
    print("\n📈 Model Evaluation Results:")
    print("============================")
    print(f"Total test cases: {num_cases}")
    print(f"Successful runs: {successful_runs}")
    print(f"Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"Average error: ${avg_error:.2f}")
    print(f"Maximum error: ${max_error:.2f}")
    print(f"\n🎯 Model Score: {score:.2f} (lower is better)")

    # Provide feedback based on exact matches
    print("\n💡 Performance Feedback:")
    if exact_matches == num_cases:
        print("🏆 PERFECT SCORE! The model has captured the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! The model is very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! The model has captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! The model understands some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")

    # Show top 5 highest error cases
    if exact_matches < num_cases:
        print("\n🔍 Top 5 Highest Error Cases:")
        high_error_cases = sorted(results, key=lambda x: x["error"], reverse=True)[:5]
        for case in high_error_cases:
            print(
                f"\nCase {case['case_num']}: {case['trip_duration']} days, "
                f"{case['miles_traveled']} miles, ${case['receipts_amount']:.2f} receipts"
            )
            print(
                f"  Expected: ${case['expected']:.2f}, Got: ${case['actual']:.2f}, "
                f"Error: ${case['error']:.2f}"
            )

    return {
        "score": score,
        "exact_matches": exact_matches,
        "close_matches": close_matches,
        "avg_error": avg_error,
        "max_error": max_error,
        "max_error_case": max_error_case,
    }


def main():
    # Load the data
    print("Loading data...")
    df = load_data()

    # Explore the data
    explore_data(df)

    # Find optimal tree depth
    optimal_depth = optimize_tree_depth(df)

    # Train the decision tree with optimal depth
    print(f"\nTraining decision tree with depth {optimal_depth}...")
    dt, X_train, X_test, y_train, y_test, y_test_pred = train_decision_tree(
        df, max_depth=optimal_depth
    )

    # Analyze feature importance
    feature_names = [
        "trip_duration_days",
        "miles_traveled",
        "total_receipts_amount",
        "miles_per_day",
        "is_efficient_trip",
        "daily_spending",
        "is_optimal_spending",
        "is_sweet_spot_trip",
        "is_five_day_trip",
        "mileage_tier",
        "receipts_per_day",
        "is_low_receipts",
        "is_high_receipts",
        "trip_type",
    ]
    analyze_feature_importance(dt, feature_names)

    # Display tree rules
    display_tree_rules(dt)

    # Plot predictions vs actual
    plot_predictions_vs_actual(y_test, y_test_pred)

    # Evaluate model performance
    print("\nEvaluating model performance...")
    evaluation_results = evaluate_model(dt, df)

    # Example predictions
    print("\nExample Predictions:")
    examples = [(3, 93, 1.42), (1, 55, 3.6), (5, 130, 306.9), (14, 958, 1727.76)]

    for trip_days, miles, receipts in examples:
        prediction = predict_new_case(dt, trip_days, miles, receipts)
        print(
            f"Trip: {trip_days} days, {miles} miles, ${receipts:.2f} receipts -> Predicted: ${prediction:.2f}"
        )

    return dt, df, evaluation_results


if __name__ == "__main__":
    model, data, evaluation_results = main()
