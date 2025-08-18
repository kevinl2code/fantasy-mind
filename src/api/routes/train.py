from fastapi import APIRouter, HTTPException
from src.api.schemas.train import PredictRBRequest, PredictRBResponse, TrainRBResponse
from src.ml.training.rb_model_trainer import RBModelTrainer
from src.ml.utils.model_storage import ModelStorage
from src.ml.utils.rb_prediction import predict_rb_fpts

router = APIRouter()


# api/routes/train.py
@router.post("/rb", response_model=TrainRBResponse)
async def train_rb_model():
    try:
        trainer = RBModelTrainer()
        results = trainer.train_rb_model()  # Always uses Snowflake

        # Save model
        storage = ModelStorage()
        model_path = storage.save_rb_model(results['model'])

        return TrainRBResponse(
            status="success",
            model_type="running_back",
            model_path=model_path,
            train_r2=results['train_r2'],
            test_r2=results['test_r2'],
            feature_importance=results['feature_importance']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/rb", response_model=PredictRBResponse)
async def predict_rb_fantasy_points(
        request: PredictRBRequest
):
    try:
        # Load latest RB model
        storage = ModelStorage()
        model = storage.load_rb_model()

        # Make prediction
        prediction = predict_rb_fpts(
            model,
            request.age,
            request.off_line_rank,
            request.games_vs_top_ten,
            request.games_vs_bottom_ten,
            request.opponent_avg_madden_rating
        )

        return {
            "position": "running_back",
            "predicted_fpts": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))