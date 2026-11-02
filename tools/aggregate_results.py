"""Mean and standard deviation of metrics over repeated runs.

python tools/aggregate_results.py outputs/seeds/imagenet1k/seed*/results.json
"""

import argparse
import json

import numpy as np

SKIP = {"step", "epoch", "num_samples", "diverged", "best", "avg_step_time_s", "peak_memory_gb"}


def main():
    parser = argparse.ArgumentParser(description="Aggregate results.json files over seeds")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    results = [json.load(open(p, encoding="utf-8")) for p in args.files]
    done = [r for r in results if not r.get("diverged")]
    print(f"{len(results)} runs, {len(results) - len(done)} diverged")
    if not done:
        return
    keys = [k for k, v in done[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in SKIP]
    for key in keys:
        values = np.array([r[key] for r in done if key in r], dtype=float)
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{key:24s} {values.mean():.2f} ± {std:.2f}   (runs: {', '.join(f'{v:.2f}' for v in values)})")


if __name__ == "__main__":
    main()
