import pandas as pd


class RBFeatureEngineer:
    """Feature engineering specifically for Running Backs"""

    def __init__(self):
        self.feature_columns = [
            'age',
            'off_line_rank',
            'games_vs_top_ten',
            'games_vs_bottom_ten',
            'opponent_avg_madden_rating',
            'tough_schedule_ratio',
            'is_prime_age',
            'is_rookie_contract'
        ]

    def prepare_features(self, df):
        """
        Prepare features for RB model

        Args:
            df: DataFrame with raw data

        Returns:
            DataFrame with cleaned and engineered features
        """
        df = df.copy()

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
        features_df['yards_per_td'] = features_df['yards'] / (features_df['touchdowns'] + 1)
        features_df['tough_schedule_ratio'] = features_df['games_vs_top_ten'] / (
                features_df['games_vs_top_ten'] + features_df['games_vs_bottom_ten'])

        # Age-related features (RB-specific prime years)
        features_df['is_prime_age'] = ((features_df['age'] >= 24) & (features_df['age'] <= 28)).astype(int)
        features_df['is_rookie_contract'] = (features_df['age'] <= 25).astype(int)

        return features_df

    def create_single_prediction_features(self, age, off_line_rank, games_vs_top_ten,
                                          games_vs_bottom_ten, opponent_avg_madden_rating):
        """Create feature array for single prediction"""
        # Calculate engineered features
        tough_schedule_ratio = games_vs_top_ten / (games_vs_top_ten + games_vs_bottom_ten)
        is_prime_age = 1 if 24 <= age <= 28 else 0
        is_rookie_contract = 1 if age <= 25 else 0

        return [
            age,
            off_line_rank,
            games_vs_top_ten,
            games_vs_bottom_ten,
            opponent_avg_madden_rating,
            tough_schedule_ratio,
            is_prime_age,
            is_rookie_contract
        ]