"""`uv run causalads <stage>` runs one stage; `uv run causalads all` runs everything."""
from __future__ import annotations

import argparse
import importlib
import time
import warnings

STAGES = ["experiment", "prediction", "observational", "uplift", "did", "report"]


def main() -> None:
    parser = argparse.ArgumentParser(prog="causalads")
    parser.add_argument("stage", choices=STAGES + ["all", "data"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    from causalads import data, io
    io.style()
    if args.stage == "data":
        df = data.load()
        print(data.validate_raw(__import__("pandas").read_csv(data.C.DATA_RAW / "hillstrom.csv")))
        print(data.arm_summary())
        print(data.splits().value_counts())
        print(df[data.FEATURES].describe().T)
        return
    for stage in STAGES if args.stage == "all" else [args.stage]:
        t0 = time.time()
        print(f"== {stage} ==", flush=True)
        importlib.import_module(f"causalads.{stage}").run()
        print(f"== {stage} done in {time.time() - t0:.1f}s ==", flush=True)
