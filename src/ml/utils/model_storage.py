# ml/utils/model_storage.py
import joblib
import json
from datetime import datetime
import os


class ModelStorage:
    def __init__(self):
        # Use persistent disk path on Render, fallback to local for development
        if os.getenv('RENDER'):
            self.base_path = "/var/data/models/"  # This will exist after Render mounts the disk
        else:
            self.base_path = "models/"  # Local development

        os.makedirs(self.base_path, exist_ok=True)
        print(f"Model storage initialized at: {self.base_path}")

    def save_rb_model(self, model, model_name="running_back_predictor"):
        """Save RB model to persistent storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save timestamped version (for version history)
        timestamped_path = f"{self.base_path}{model_name}_{timestamp}.pkl"
        joblib.dump(model, timestamped_path)

        # Save latest version (for easy loading)
        latest_path = f"{self.base_path}{model_name}_latest.pkl"
        joblib.dump(model, latest_path)

        # Save metadata
        metadata = {
            "model_type": "running_back",
            "model_name": model_name,
            "timestamp": timestamp,
            "timestamped_path": timestamped_path,
            "latest_path": latest_path,
            "feature_columns": model.feature_columns,
            "storage_location": "persistent_disk" if os.getenv('RENDER') else "local"
        }

        metadata_path = f"{self.base_path}{model_name}_latest_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Model saved to persistent disk: {latest_path}")
        return latest_path

    def load_rb_model(self, model_name="running_back_predictor"):
        """Load the latest RB model from persistent storage"""
        latest_path = f"{self.base_path}{model_name}_latest.pkl"

        if not os.path.exists(latest_path):
            available_models = self.list_saved_models()
            raise FileNotFoundError(
                f"No RB model found at {latest_path}. "
                f"Available models: {available_models}"
            )

        print(f"📂 Loading model from persistent disk: {latest_path}")
        return joblib.load(latest_path)

    def model_exists(self, model_name="running_back_predictor"):
        """Check if a model exists"""
        latest_path = f"{self.base_path}{model_name}_latest.pkl"
        return os.path.exists(latest_path)

    def list_saved_models(self):
        """List all saved model files"""
        if not os.path.exists(self.base_path):
            return []

        models = []
        for file in os.listdir(self.base_path):
            if file.endswith('.pkl'):
                models.append(file)
        return models

    def get_model_info(self, model_name="running_back_predictor"):
        """Get metadata about the saved model"""
        metadata_path = f"{self.base_path}{model_name}_latest_metadata.json"

        if not os.path.exists(metadata_path):
            return {"error": "No metadata found", "model_exists": self.model_exists(model_name)}

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Add file size info
        if self.model_exists(model_name):
            latest_path = f"{self.base_path}{model_name}_latest.pkl"
            file_size_mb = os.path.getsize(latest_path) / (1024 * 1024)
            metadata["file_size_mb"] = round(file_size_mb, 2)

        return metadata