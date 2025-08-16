import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class RunningBackPredictor:
    def __init__(self):
        self.model = SGDRegressor(
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            alpha=0.01
        )
        self.scaler = StandardScaler()
        self.is_trained = False
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

    def train(self, X_train, y_train):
        """Train the model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        return self.model

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {
            'mse': mse,
            'r2': r2,
            'predictions': predictions
        }

    def get_feature_importance(self):
        """Get feature importance (coefficients)"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        return dict(zip(self.feature_columns, self.model.coef_))