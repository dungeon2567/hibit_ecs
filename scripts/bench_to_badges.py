#!/usr/bin/env python3
"""Convert Google Benchmark JSON into shields.io endpoint badges + a SUMMARY.md table.

Usage: bench_to_badges.py <bench.json> <output_dir>

Output layout (under <output_dir>):
  badges/<slug>.json   — one shields endpoint JSON per benchmark median
  SUMMARY.md           — markdown table of medians
"""
import json
import pathlib
import re
import sys


def slug(name: str) -> str:
    s = re.sub(r"^BM_", "", name)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()
    return s


def fmt(value: float, unit: str) -> str:
    if unit == "ns":
        if value >= 1e6:
            return f"{value / 1e6:.2f} ms"
        if value >= 1e3:
            return f"{value / 1e3:.2f} µs"
        return f"{value:.0f} ns"
    if unit == "us":
        if value >= 1e3:
            return f"{value / 1e3:.2f} ms"
        return f"{value:.2f} µs"
    if unit == "ms":
        if value >= 1e3:
            return f"{value / 1e3:.2f} s"
        return f"{value:.2f} ms"
    return f"{value:.2f} {unit}"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    badges_dir = dst / "badges"
    badges_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text())

    medians = []
    for b in data.get("benchmarks", []):
        name = b["name"]
        if not name.endswith("_median"):
            continue
        base = name[: -len("_median")]
        s = slug(base)
        msg = fmt(b["real_time"], b["time_unit"])
        badge = {
            "schemaVersion": 1,
            "label": base,
            "message": msg,
            "color": "blue",
        }
        (badges_dir / f"{s}.json").write_text(json.dumps(badge))
        medians.append((base, msg))

    if not medians:
        print("No '_median' rows found — did you pass --benchmark_repetitions=N --benchmark_report_aggregates_only=true?", file=sys.stderr)
        return 1

    lines = ["# Benchmark summary", "", "| Bench | Median |", "| --- | --- |"]
    for name, msg in medians:
        lines.append(f"| `{name}` | {msg} |")
    (dst / "SUMMARY.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(medians)} badges + SUMMARY.md to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
