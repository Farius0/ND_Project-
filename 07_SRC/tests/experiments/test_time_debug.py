# ======================================================
# ====[ TimerManager Demo – Profiling + Logging ]=====
# ======================================================
import warnings; warnings.filterwarnings('ignore')
import time, json, pandas as pd
from pathlib import Path

import sys
sys.path.append(str(Path.cwd().parent))

from utils.logger import get_logger, get_debug_logger
from utils.decorators import TimerManager, safe_timer, safe_timer_and_debug

# ====[ Setup ]====
timer = TimerManager()
logger = get_logger(name="timing_demo")

# ====[ Decorated functions ]====
@timer.decorator(name="task_add")
def add():
    time.sleep(0.3)

@timer.decorator(name="task_mul")
def mul():
    time.sleep(0.1)

# ====[ Run tasks ]====
for _ in range(2): add()
for _ in range(3): mul()

# ====[ Method 1: Console Summary ]====
timer.summary()

# ====[ Method 2: Export to Dict / JSON ]====
data = timer.to_dict()
print("\nJSON Export:")
print(json.dumps(data, indent=2))

# ====[ Method 3: DataFrame / Notebook Table ]====
df = pd.DataFrame(timer.to_list(), columns=["Name", "Calls", "Total (s)", "Avg (s)"])
print("\nDataFrame Table:")
print(df)

# ====[ Method 4: Logging to File ]====
timer.to_log(logger=logger)

# ===========================================================
# ====[ Demo: safe_timer & safe_timer_and_debug usage ]=====
# ===========================================================

# ====[ Setup custom loggers ]====
info_logger = get_logger(name="safe_timer_info")
debug_logger = get_debug_logger(name="safe_timer_debug")

# ====[ Basic function wrapped with safe_timer ]====
@safe_timer(name="sleep_short", info_logger=info_logger)
def short_task():
    time.sleep(0.2)

# ====[ Function with error and warning ]====
@safe_timer(name="unstable", info_logger=info_logger, warning_message="Input might be unstable")
def risky_task():
    time.sleep(0.1)
    raise RuntimeError("Unexpected failure")

# ====[ Function wrapped with full debug ]====
@safe_timer_and_debug(
    name="long_task",
    log_inputs=True,
    log_output=True,
    info_logger=info_logger,
    debug_logger=debug_logger
)
def compute(a, b):
    time.sleep(0.15)
    return a * b + 42

# ====[ Execute functions ]====
short_task()
result = compute(3, 5)
print("Result:", result)

try:
    risky_task()
except Exception as e:
    print("Handled exception:", e)

