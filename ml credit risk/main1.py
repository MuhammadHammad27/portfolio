"""Legacy script redirecting to the modern ML training pipeline in ml/train.py."""
import sys
from ml.train import run_training, parse_args

if __name__ == "__main__":
    print("[INFO] Redirecting to production pipeline in ml/train.py ...")
    args = parse_args()
    run_training(args)
