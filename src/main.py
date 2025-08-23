import os
import sys
from src.helpers.loggers import log

def main():
    run_mode = os.getenv("RUN_MODE", "train")

    if run_mode == 'train':
        from src.train import start_training
        from src.test.test_model import run_test_training
        start_training()

        log("Training completed.")

        log("Starting evaluation...")

        run_test_training(mode=os.getenv("TEST_MODE","force_think"))

        log("Evaluation completed. Finish the program.")
        return
    elif run_mode == "test-training":
        from src.test.test_model import run_test_training
        run_test_training(mode=os.getenv("TEST_MODE","force_think"))
        return
    elif run_mode == "test-merging":
        from src.test.test_model import run_test_merging
        run_test_merging(mode=os.getenv("TEST_MODE","force_think"))
        return
    elif run_mode == "test-gguf":
        from src.test.test_model import run_test_gguf
        run_test_gguf(mode=os.getenv("TEST_MODE","force_think"))
        return
    else:
        # error
        print(f"Unknown RUN_MODE: {run_mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()