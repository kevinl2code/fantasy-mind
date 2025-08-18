# test_training.py (in root directory)
from dotenv import load_dotenv
load_dotenv()

from src.ml.training.rb_model_trainer import RBModelTrainer

if __name__ == "__main__":
    trainer = RBModelTrainer()
    results = trainer.train_rb_model()
    print("Training completed successfully!")