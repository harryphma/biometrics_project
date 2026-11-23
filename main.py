#!/usr/bin/env python3
"""Command-line interface for SoF manifest preparation and biometric experiments."""

import argparse
import json
from pathlib import Path
from biometrics.data import prepare_sof
from biometrics.experiments import run


def main():
    """Parse command-line options and dispatch manifest preparation or configured experiments."""
    p = argparse.ArgumentParser(
        description="CAP4103 SoF + CMU biometric verification experiments"
    )
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser(
        "prepare-sof",
        help="Build a manifest from manually downloaded/extracted originals",
    )
    f.add_argument("--images", required=True)
    f.add_argument("--metadata", required=True)
    f.add_argument("--out", default="data/sof_manifest.csv")
    f.add_argument("--gap", type=int, default=1)
    r = sub.add_parser("run", help="Run RQ1, RQ2, RQ3 on the supplied real datasets")
    r.add_argument("--config", default="config/default.json")
    r.add_argument("--output")
    r.add_argument(
        "--seed", type=int, help="Override with one seed for a quick first run"
    )
    r.add_argument("--no-plots", action="store_true")
    a = p.parse_args()
    if a.command == "prepare-sof":
        m = prepare_sof(a.images, a.metadata, a.out, gap=a.gap)
        print(f"{len(m)} samples -> {a.out}")
    else:
        cfg = json.loads(Path(a.config).read_text())
        if a.output:
            cfg["output"] = a.output
        if a.seed is not None:
            cfg["seeds"] = [a.seed]
        if a.no_plots:
            cfg["plots"] = False
        run(cfg)


if __name__ == "__main__":
    main()
