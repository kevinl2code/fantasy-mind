from pydantic import BaseModel
from typing import Dict, Optional

class PredictRBRequest(BaseModel):
    age: int
    off_line_rank: int
    games_vs_top_ten: int
    games_vs_bottom_ten: int
    opponent_avg_madden_rating: float

class PredictRBResponse(BaseModel):
    position: str
    predicted_fpts: float

class TrainRBResponse(BaseModel):
    status: str
    model_type: str
    model_path: str
    train_r2: float
    test_r2: float
    feature_importance: Dict[str, float]