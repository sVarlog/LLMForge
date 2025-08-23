import os
import sys

def main():
    run_mode = os.getenv("RUN_MODE", "train")

    if run_mode == "test-training":
        from src.test.test_model import run_test_training
        run_test_training(mode=os.getenv("TEST_MODE","force_think"))
        return

    if run_mode == "test-merging":
        from src.test.test_model import run_test_merging
        run_test_merging(mode=os.getenv("TEST_MODE","force_think"))
        return

    if run_mode == "test-gguf":
        from src.test.test_model import run_test_gguf
        run_test_gguf(mode=os.getenv("TEST_MODE","force_think"))
        return

    # else: your normal train/server entry
    from src.train import main as train_main
    train_main()

if __name__ == "__main__":
    main()