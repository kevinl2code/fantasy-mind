# src/ml/training/rb_model_trainer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from src.ml.models.running_back_predictor import RunningBackPredictor
from src.ml.preprocessing.data_cleaner import DataCleaner
from src.ml.preprocessing.rb_feature_engineer import RBFeatureEngineer
from src.ml.utils.model_storage import ModelStorage
from src.database.snowflake import get_snowflake_connection


class RBModelTrainer:
    def __init__(self):
        self.data_cleaner = DataCleaner()
        self.rb_feature_engineer = RBFeatureEngineer()
        self.model_storage = ModelStorage()

    def train_rb_model(self, df=None):
        """
        Train a Running Back FPTS prediction model

        Args:
            df: DataFrame with data (optional, if not provided loads from Snowflake)

        Returns:
            Dictionary containing model results and performance metrics
        """
        # Load data
        print("Loading data...")

        if df is not None:
            data_df = df.copy()
        else:
            data_df = self._load_from_snowflake()

        print(f"Loaded {len(data_df)} rows")

        # Clean data
        print("Cleaning data...")
        data_df = self.data_cleaner.clean_dataframe(data_df)

        # Prepare features
        print("Preparing RB features...")
        features_df = self.rb_feature_engineer.prepare_features(data_df)

        # Prepare X (features) and y (target)
        X = features_df[self.rb_feature_engineer.feature_columns]
        y = features_df['fpts']

        print(f"Features: {self.rb_feature_engineer.feature_columns}")
        print(f"Target: FPTS (range: {y.min():.1f} - {y.max():.1f})")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train model
        print("Training Running Back Predictor...")
        model = RunningBackPredictor()
        model.train(X_train, y_train)

        # Evaluate
        train_results = model.evaluate(X_train, y_train)
        test_results = model.evaluate(X_test, y_test)

        print("\n=== Model Performance ===")
        print(f"Training MSE: {train_results['mse']:.2f}")
        print(f"Test MSE: {test_results['mse']:.2f}")
        print(f"Training R²: {train_results['r2']:.3f}")
        print(f"Test R²: {test_results['r2']:.3f}")

        # Feature importance
        print("\n=== Feature Importance ===")
        feature_importance = model.get_feature_importance()
        for feature, coef in feature_importance.items():
            print(f"{feature}: {coef:.3f}")

        # Sample predictions
        print("\n=== Sample Predictions ===")
        comparison_df = pd.DataFrame({
            'Actual': y_test.iloc[:5].values,
            'Predicted': test_results['predictions'][:5],
            'Difference': y_test.iloc[:5].values - test_results['predictions'][:5]
        })
        print(comparison_df.round(1))

        return {
            'model': model,
            'rb_feature_engineer': self.rb_feature_engineer,
            'train_mse': train_results['mse'],
            'test_mse': test_results['mse'],
            'train_r2': train_results['r2'],
            'test_r2': test_results['r2'],
            'feature_importance': feature_importance,
            'test_data': (X_test, y_test, test_results['predictions'])
        }

    def _load_from_snowflake(self):
        """Load training data from Snowflake table"""
        print("Connecting to Snowflake...")
        conn = get_snowflake_connection()

        try:
            query = """
            SELECT 
                YEAR,
                FFRANK,
                NAME,
                AGE,
                TEAM,
                TEAMID,
                OFF_LINE_RANK,
                YARDS,
                TD,
                FPTS,
                GAMES_VS_TOP_TEN,
                GAMES_VS_BOTTOM_TEN,
                OPPONENT_AVG_MADDEN_RATING
            FROM FANTASY_MIND_DATA.STAGING.YEARLY_TOP_20_RB
            ORDER BY YEAR, FFRANK
            """

            print("Executing query...")
            df = pd.read_sql(query, conn)
            print(f"Successfully loaded {len(df)} rows from Snowflake")

            return df

        except Exception as e:
            print(f"Error loading from Snowflake: {e}")
            raise
        finally:
            conn.close()
            print("Snowflake connection closed")

    def get_data_summary(self):
        """Get a summary of the available training data"""
        try:
            conn = get_snowflake_connection()

            summary_query = """
            SELECT 
                COUNT(*) as total_records,
                MIN(YEAR) as earliest_year,
                MAX(YEAR) as latest_year,
                COUNT(DISTINCT NAME) as unique_players,
                COUNT(DISTINCT TEAM) as unique_teams,
                ROUND(AVG(FPTS), 2) as avg_fpts,
                ROUND(MIN(FPTS), 2) as min_fpts,
                ROUND(MAX(FPTS), 2) as max_fpts
            FROM FANTASY_MIND_DATA.STAGING.YEARLY_TOP_20_RB
            """

            summary_df = pd.read_sql(summary_query, conn)
            return summary_df.to_dict('records')[0]

        except Exception as e:
            print(f"Error getting data summary: {e}")
            return None
        finally:
            conn.close()