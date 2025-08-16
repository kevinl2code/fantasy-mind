import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import re


def clean_yards_column(yards_str):
    """Clean the Yards column by removing commas and converting to int"""
    if isinstance(yards_str, str):
        # Remove commas and convert to int
        return int(yards_str.replace(',', ''))
    return yards_str


def prepare_features(df):
    """
    Prepare features for the model

    Args:
        df: DataFrame with raw data

    Returns:
        DataFrame with cleaned and engineered features
    """
    df = df.copy()

    # Clean the Yards column
    df['Yards'] = df['Yards'].apply(clean_yards_column)

    # Create feature columns for modeling
    features_df = pd.DataFrame({
        'year': df['Year'],
        'age': df['Age'],
        'off_line_rank': df['off_line_rank'],
        'yards': df['Yards'],
        'touchdowns': df['TD'],
        'games_vs_top_ten': df['games_vs_top_ten'],
        'games_vs_bottom_ten': df['games_vs_bottom_ten'],
        'opponent_avg_madden_rating': df['opponent_avg_madden_rating'],
        'fpts': df['FPTS']  # target variable
    })

    # Create additional features
    features_df['yards_per_td'] = features_df['yards'] / (features_df['touchdowns'] + 1)  # +1 to avoid division by zero
    features_df['tough_schedule_ratio'] = features_df['games_vs_top_ten'] / (
                features_df['games_vs_top_ten'] + features_df['games_vs_bottom_ten'])

    # Age-related features
    features_df['is_prime_age'] = ((features_df['age'] >= 24) & (features_df['age'] <= 28)).astype(
        int)  # RB prime years
    features_df['is_rookie_contract'] = (features_df['age'] <= 25).astype(int)  # Likely on rookie deal

    return features_df


def train_fpts_model(csv_file_path):
    """
    Train an SGD regressor to predict FPTS

    Args:
        csv_file_path: Path to the CSV file

    Returns:
        Dictionary containing model, scaler, and performance metrics
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} rows")

    # Prepare features
    print("Preparing features...")
    features_df = prepare_features(df)

    # Define feature columns (excluding target) - features available for prediction
    feature_columns = [
        'age',
        'off_line_rank',
        'games_vs_top_ten',
        'games_vs_bottom_ten',
        'opponent_avg_madden_rating',
        'tough_schedule_ratio',
        'is_prime_age',
        'is_rookie_contract'
    ]

    # Prepare X (features) and y (target)
    X = features_df[feature_columns]
    y = features_df['fpts']

    print(f"Features: {feature_columns}")
    print(f"Target: FPTS (range: {y.min():.1f} - {y.max():.1f})")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SGD Regressor
    print("Training SGD Regressor...")
    model = SGDRegressor(
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        alpha=0.01  # regularization strength
    )

    model.fit(X_train_scaled, y_train)

    # Make predictions
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    print("\n=== Model Performance ===")
    print(f"Training MSE: {train_mse:.2f}")
    print(f"Test MSE: {test_mse:.2f}")
    print(f"Training R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")

    # Feature importance (coefficients)
    print("\n=== Feature Importance ===")
    for feature, coef in zip(feature_columns, model.coef_):
        print(f"{feature}: {coef:.3f}")

    # Show some predictions vs actual
    print("\n=== Sample Predictions ===")
    comparison_df = pd.DataFrame({
        'Actual': y_test.iloc[:5],
        'Predicted': test_predictions[:5],
        'Difference': y_test.iloc[:5] - test_predictions[:5]
    })
    print(comparison_df.round(1))

    return {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_data': (X_test, y_test, test_predictions)
    }


def predict_fpts(model_data, age, off_line_rank, games_vs_top_ten, games_vs_bottom_ten, opponent_avg_madden_rating):
    """
    Make a single FPTS prediction

    Args:
        model_data: Dictionary returned from train_fpts_model
        age: Player age
        off_line_rank: Offensive line ranking (1-32, lower is better)
        games_vs_top_ten: Games against top 10 defenses
        games_vs_bottom_ten: Games against bottom 10 defenses
        opponent_avg_madden_rating: Average opponent Madden rating

    Returns:
        Predicted FPTS
    """
    # Calculate engineered features
    tough_schedule_ratio = games_vs_top_ten / (games_vs_top_ten + games_vs_bottom_ten)
    is_prime_age = 1 if 24 <= age <= 28 else 0
    is_rookie_contract = 1 if age <= 25 else 0

    # Create feature array
    features = np.array([[
        age,
        off_line_rank,
        games_vs_top_ten,
        games_vs_bottom_ten,
        opponent_avg_madden_rating,
        tough_schedule_ratio,
        is_prime_age,
        is_rookie_contract
    ]])

    # Scale features
    features_scaled = model_data['scaler'].transform(features)

    # Make prediction
    prediction = model_data['model'].predict(features_scaled)[0]

    return round(prediction, 1)


if __name__ == "__main__":
    # Train the model
    csv_path = "yearly_top_20.csv"
    model_data = train_fpts_model(csv_path)

    print("\n=== Example Prediction ===")
    # Example prediction for a 25-year-old RB with good offensive line
    predicted_fpts = predict_fpts(
        model_data,
        age=25,
        off_line_rank=8,  # Good offensive line
        games_vs_top_ten=4,
        games_vs_bottom_ten=6,
        opponent_avg_madden_rating=79.0
    )
    print(f"Predicted FPTS for 25yr old RB with rank 8 O-line: {predicted_fpts}")

    # Compare with older RB and worse O-line
    predicted_fpts_old = predict_fpts(
        model_data,
        age=30,
        off_line_rank=20,  # Poor offensive line
        games_vs_top_ten=4,
        games_vs_bottom_ten=6,
        opponent_avg_madden_rating=79.0
    )
    print(f"Predicted FPTS for 30yr old RB with rank 20 O-line: {predicted_fpts_old}")