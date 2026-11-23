"""Command-line entry points for the Week 1 reproducible vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .datasets import generate_events, load_events, save_events


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate event recordings and save diagnostic figures.")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="generate, reload and visualize a deterministic synthetic recording")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--seed", type=int, default=7)
    view = commands.add_parser("view", help="load a CSV/NPZ recording and save a figure plus JSON summary")
    view.add_argument("input", type=Path)
    view.add_argument("--width", type=int, required=True, help="sensor width in pixels")
    view.add_argument("--height", type=int, required=True, help="sensor height in pixels")
    view.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"), required=True)
    view.add_argument("--output", type=Path, default=Path("outputs/event_stream.png"))
    for command in (demo, view):
        command.add_argument("--start-ms", type=float, default=0, help="start of four adjacent accumulation windows (default: 0)")
        command.add_argument("--window-ms", type=float, default=100, help="duration of each accumulation window (default: 100)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if not np.isfinite(args.start_ms) or args.start_ms < 0:
            raise ValueError("--start-ms must be finite and nonnegative")
        if not np.isfinite(args.window_ms) or args.window_ms <= 0:
            raise ValueError("--window-ms must be finite and positive")
        if args.command == "demo":
            source = args.output_dir / "synthetic_events.npz"
            output = args.output_dir / "event_stream.png"
            save_events(generate_events(seed=args.seed), source)
            width, height, unit = 128, 96, "us"
            title = "Synthetic moving edges · ingestion demo, not VO results"
        else:
            source, output = args.input, args.output
            width, height, unit = args.width, args.height, args.timestamp_unit
            title = f"Event-stream diagnostics · {source.name}"
        if output.suffix.lower() not in (".png", ".svg", ".pdf"):
            raise ValueError("--output must end in .png, .svg or .pdf")
        events = load_events(source, width=width, height=height, timestamp_unit=unit)
        from .visualization.plots import save_report
        import matplotlib

        summary = save_report(events, output, start=args.start_ms / 1000, duration=args.window_ms / 1000, title=title)
        summary.update({
            "schema_version": 1,
            "source_path": str(source),
            "source_sha256": _sha256(source),
            "figure_path": str(output),
            "data_kind": "synthetic" if args.command == "demo" else "user_supplied",
            "synthetic_seed": args.seed if args.command == "demo" else None,
            "command": ["adaptive-event-vo", *(argv if argv is not None else sys.argv[1:])],
            "environment": {
                "python": platform.python_version(), "platform": platform.platform(),
                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "adaptive_event_vo": __version__,
            },
        })
        summary_path = output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Validated {len(events):,} events ({width} x {height}, {summary['observed_span_s']:.6f} s observed span)")
        print(f"Figure:  {output}\nSummary: {summary_path}")
        return 0
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
