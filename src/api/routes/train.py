from fastapi import APIRouter, HTTPException
from ml.training.rb_model_trainer import RBModelTrainer
from ml.utils.model_storage import ModelStorage
from ml.utils.rb_prediction import predict_rb_fpts

router = APIRouter()


@router.post("/train/rb")
async def train_rb_model(csv_path: str = "yearly_top_20.csv"):
    try:
        trainer = RBModelTrainer()
        results = trainer.train_rb_model(csv_file_path=csv_path)

        # Save model
        storage = ModelStorage()
        model_path = storage.save_rb_model(results['model'])

        return {
            "status": "success",
            "model_type": "running_back",
            "model_path": model_path,
            "train_r2": results['train_r2'],
            "test_r2": results['test_r2'],
            "feature_importance": results['feature_importance']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/rb")
async def predict_rb_fantasy_points(
        age: int,
        off_line_rank: int,
        games_vs_top_ten: int,
        games_vs_bottom_ten: int,
        opponent_avg_madden_rating: float
):
    try:
        # Load latest RB model
        storage = ModelStorage()
        model = storage.load_rb_model()

        # Make prediction
        prediction = predict_rb_fpts(
            model, age, off_line_rank, games_vs_top_ten,
            games_vs_bottom_ten, opponent_avg_madden_rating
        )

        return {
            "position": "running_back",
            "predicted_fpts": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))