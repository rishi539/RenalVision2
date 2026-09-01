import os
import sys
from pathlib import Path

# Add project root directory to python path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_model import run_full_training


if __name__ == "__main__":
    print("==========================================")
    print("🚀 Starting 4-Class Kidney Disease Model Training")
    print("==========================================")
    run_full_training()
