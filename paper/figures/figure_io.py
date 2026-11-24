"""Common command-line options for the paper figures."""
import argparse
import json
from pathlib import Path


def figure_paths(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[2] / "results",
                        help="directory containing the exported experiment results")
    parser.add_argument("--output", type=Path, default=Path.cwd(),
                        help="directory for the rendered PDF")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args.results, args.output


def load(results, filename):
    return json.loads((results / filename).read_text(encoding="utf8"))
