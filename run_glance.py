"""Command-line runner for GLANCE training and evaluation."""

from __future__ import annotations

import argparse
import json

import train_glance as experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root")
    parser.add_argument("--channel-order", required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run checks only; do not create checkpoints or train.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Explicitly start the 200-epoch GLANCE training run.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.run or args.preflight_only:
        report = experiment.preflight(args)
        print(json.dumps(report, indent=2, default=str), flush=True)

        if not args.run:
            print(
                "Preflight only: training was not started. "
                "Pass --run explicitly to launch.",
                flush=True,
            )
        return

    summary = experiment.train(args)
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
