from src.training_config import (
    FINAL_LOG_FH, 
    DEF_LOG_PREFIX, 
    TEE_ACTIVE, 
    DEBUG, 
    DEF_DBG_PREFIX
)

def _write_sink(s: str):
    try:
        if FINAL_LOG_FH:
            FINAL_LOG_FH.write(s + "\n")
            FINAL_LOG_FH.flush()
    except Exception:
        pass

def log(msg):
    s = f"\n{DEF_LOG_PREFIX}{msg}\n{'=' * 60}"
    print(s)
    # avoid double-writing: when tee is active, print already goes to file
    if not TEE_ACTIVE:
        _write_sink(s)


def debug(msg):
    if DEBUG:
        s = f"\n{DEF_DBG_PREFIX}{msg}\n{'-' * 60}"
        print(s)
        if not TEE_ACTIVE:
            _write_sink(s)