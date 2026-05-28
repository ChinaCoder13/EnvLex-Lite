from __future__ import annotations

import argparse
import os
import subprocess
import sys


def call(script: str, data: str, out: str, extra=None):
    cmd = [sys.executable, os.path.join("code", script), "--data", data, "--out", out]
    if extra:
        cmd.extend(extra)
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--tables", default="tables")
    parser.add_argument("--figures", default="figures")
    args = parser.parse_args()

    call("run_baselines.py", args.data, args.out)
    call("train_model.py", args.data, args.out)
    call("run_ablation.py", args.data, args.out)
    subprocess.check_call([sys.executable, os.path.join("code", "make_tables.py"), "--data", args.data, "--out", args.out, "--tables", args.tables])
    subprocess.check_call([sys.executable, os.path.join("code", "make_figures.py"), "--data", args.data, "--out", args.out, "--figures", args.figures])


if __name__ == "__main__":
    main()
