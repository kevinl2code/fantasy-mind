# ml/utils/prediction.py
import numpy as np
from ml.preprocessing.rb_feature_engineer import RBFeatureEngineer


def predict_rb_fpts(model, age, off_line_rank, games_vs_top_ten, games_vs_bottom_ten, opponent_avg_madden_rating):
    """
    Make a single RB FPTS prediction

    Args:
        model: Trained RunningBackPredictor model
        age: Player age
        off_line_rank: Offensive line ranking (1-32, lower is better)
        games_vs_top_ten: Games against top 10 defenses
        games_vs_bottom_ten: Games against bottom 10 defenses
        opponent_avg_madden_rating: Average opponent Madden rating

    Returns:
        Predicted FPTS (rounded to 1 decimal place)
    """
    rb_feature_engineer = RBFeatureEngineer()

    # Create feature array using the feature engineer
    features = rb_feature_engineer.create_single_prediction_features(
        age, off_line_rank, games_vs_top_ten, games_vs_bottom_ten, opponent_avg_madden_rating
    )

    # Convert to numpy array (model expects 2D array)
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)[0]

    return round(prediction, 1)


def predict_rb_fpts_batch(model, predictions_data):
    """
    Make multiple RB FPTS predictions at once

    Args:
        model: Trained RunningBackPredictor model
        predictions_data: List of dicts with prediction parameters

    Example:
        predictions_data = [
            {
                "age": 25,
                "off_line_rank": 8,
                "games_vs_top_ten": 4,
                "games_vs_bottom_ten": 6,
                "opponent_avg_madden_rating": 79.0
            },
            {
                "age": 30,
                "off_line_rank": 20,
                "games_vs_top_ten": 6,
                "games_vs_bottom_ten": 4,
                "opponent_avg_madden_rating": 82.0
            }
        ]

    Returns:
        List of predicted FPTS values
    """
    rb_feature_engineer = RBFeatureEngineer()

    # Prepare all features
    all_features = []
    for data in predictions_data:
        features = rb_feature_engineer.create_single_prediction_features(
            data["age"],
            data["off_line_rank"],
            data["games_vs_top_ten"],
            data["games_vs_bottom_ten"],
            data["opponent_avg_madden_rating"]
        )
        all_features.append(features)

    # Convert to numpy array
    features_array = np.array(all_features)

    # Make predictions
    predictions = model.predict(features_array)

    # Round and return
    return [round(pred, 1) for pred in predictions]


def validate_prediction_inputs(age, off_line_rank, games_vs_top_ten, games_vs_bottom_ten, opponent_avg_madden_rating):
    """
    Validate inputs for RB prediction

    Returns:
        dict: {"valid": bool, "errors": list}
    """
    errors = []

    # Age validation
    if not isinstance(age, int) or age < 18 or age > 40:
        errors.append("Age must be an integer between 18 and 40")

    # Offensive line rank validation
    if not isinstance(off_line_rank, int) or off_line_rank < 1 or off_line_rank > 32:
        errors.append("Offensive line rank must be an integer between 1 and 32")

    # Games validation
    if not isinstance(games_vs_top_ten, int) or games_vs_top_ten < 0 or games_vs_top_ten > 17:
        errors.append("Games vs top ten must be an integer between 0 and 17")

    if not isinstance(games_vs_bottom_ten, int) or games_vs_bottom_ten < 0 or games_vs_bottom_ten > 17:
        errors.append("Games vs bottom ten must be an integer between 0 and 17")

    # Total games check
    total_games = games_vs_top_ten + games_vs_bottom_ten
    if total_games > 17:
        errors.append("Total games (vs top ten + vs bottom ten) cannot exceed 17")

    if total_games == 0:
        errors.append("Must have at least 1 game vs top ten or bottom ten defenses")

    # Madden rating validation
    if not isinstance(opponent_avg_madden_rating,
                      (int, float)) or opponent_avg_madden_rating < 50 or opponent_avg_madden_rating > 99:
        errors.append("Opponent average Madden rating must be a number between 50 and 99")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def get_prediction_explanation(age, off_line_rank, games_vs_top_ten, games_vs_bottom_ten, opponent_avg_madden_rating):
    """
    Provide human-readable explanation of prediction inputs

    Returns:
        dict with explanation of each factor
    """
    rb_feature_engineer = RBFeatureEngineer()

    # Calculate derived features for explanation
    tough_schedule_ratio = games_vs_top_ten / (games_vs_top_ten + games_vs_bottom_ten)
    is_prime_age = 24 <= age <= 28
    is_rookie_contract = age <= 25

    # Determine strength descriptions
    def get_oline_strength(rank):
        if rank <= 5:
            return "Elite"
        elif rank <= 10:
            return "Good"
        elif rank <= 20:
            return "Average"
        elif rank <= 28:
            return "Below Average"
        else:
            return "Poor"

    def get_schedule_difficulty(ratio):
        if ratio >= 0.6:
            return "Very Tough"
        elif ratio >= 0.4:
            return "Tough"
        elif ratio >= 0.3:
            return "Average"
        else:
            return "Easy"

    return {
        "age_analysis": {
            "age": age,
            "is_prime_age": is_prime_age,
            "is_rookie_contract": is_rookie_contract,
            "description": f"{'Prime age' if is_prime_age else 'Non-prime age'} RB, {'likely on rookie contract' if is_rookie_contract else 'veteran contract'}"
        },
        "offensive_line": {
            "rank": off_line_rank,
            "strength": get_oline_strength(off_line_rank),
            "description": f"Rank #{off_line_rank} offensive line ({get_oline_strength(off_line_rank).lower()})"
        },
        "schedule_difficulty": {
            "games_vs_top_ten": games_vs_top_ten,
            "games_vs_bottom_ten": games_vs_bottom_ten,
            "tough_schedule_ratio": round(tough_schedule_ratio, 2),
            "difficulty": get_schedule_difficulty(tough_schedule_ratio),
            "description": f"{get_schedule_difficulty(tough_schedule_ratio).lower()} schedule ({games_vs_top_ten} vs top 10, {games_vs_bottom_ten} vs bottom 10)"
        },
        "opponent_strength": {
            "avg_madden_rating": opponent_avg_madden_rating,
            "description": f"Average opponent Madden rating of {opponent_avg_madden_rating}"
        }
    }