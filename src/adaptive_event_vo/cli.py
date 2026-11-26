"""Event diagnostics and reproducible fixed-window image-motion experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
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
    baseline = commands.add_parser("baseline", help="run identical image-motion pipelines at fixed window durations")
    baseline.add_argument("input", type=Path)
    baseline.add_argument("--config", type=Path, help="JSON baseline configuration; default is 10/25/50 ms, three repetitions")
    baseline.add_argument("--manifest", type=Path, help="converted DAVIS clip manifest with dimensions, units and input hash")
    baseline.add_argument("--width", type=int)
    baseline.add_argument("--height", type=int)
    baseline.add_argument("--timestamp-unit", choices=("s", "ms", "us", "ns"))
    baseline.add_argument("--output-dir", type=Path, default=Path("outputs/baseline"))
    validation = commands.add_parser("validate-flow", help="test known synthetic count-image translations and failure cases")
    validation.add_argument("--output-dir", type=Path, default=Path("outputs/flow_validation"))
    motion_demo = commands.add_parser("motion-demo", help="run the baseline on a seeded known-translation event fixture")
    motion_demo.add_argument("--output-dir", type=Path, default=Path("outputs/motion_demo"))
    for command in (demo, view):
        command.add_argument("--start-ms", type=float, default=0, help="start of four adjacent accumulation windows (default: 0)")
        command.add_argument("--window-ms", type=float, default=100, help="duration of each accumulation window (default: 100)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in ("baseline", "validate-flow", "motion-demo"):
            return _motion_command(args, argv)
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


def _motion_command(args, argv):
    from .evaluation.baseline import BaselineConfig, run_baseline
    from .provenance import sha256_file

    if args.command == "validate-flow":
        from .evaluation.flow_validation import validate_flow
        result = validate_flow(args.output_dir)
        print(f"Known-translation validation: {'PASS' if result['passed'] else 'FAIL'}; {args.output_dir / 'validation.json'}")
        return 0 if result["passed"] else 1
    if args.command == "motion-demo":
        from .datasets.motion_fixture import translated_events
        args.output_dir.mkdir(parents=True, exist_ok=True)
        source = args.output_dir / "synthetic_motion.npz"
        save_events(translated_events(), source)
        events = load_events(source, width=128, height=96, timestamp_unit="us")
        config = BaselineConfig(window_ms=(10,), end_s=0.02)
        provenance = {"data_kind": "synthetic_count_texture", "known_displacement_px": [2, -1], "seed": 19}
    else:
        source = args.input
        config = BaselineConfig.from_dict(json.loads(args.config.read_text())) if args.config else BaselineConfig()
        manifest = json.loads(args.manifest.read_text()) if args.manifest else {}
        if args.manifest and (not isinstance(manifest, dict) or not {"events_sha256", "normalized_interval_end_s", "width", "height", "timestamp_unit"}.issubset(manifest)):
            raise ValueError("manifest is missing required clip metadata")
        if manifest and manifest.get("events_sha256") != sha256_file(source):
            raise ValueError("manifest event hash does not match input recording")
        dimensions = {}
        for key in ("width", "height", "timestamp_unit"):
            explicit = getattr(args, key)
            if explicit is not None and key in manifest and explicit != manifest[key]:
                raise ValueError(f"explicit {key} disagrees with the dataset manifest")
            dimensions[key] = explicit if explicit is not None else manifest.get(key)
        if any(value is None for value in dimensions.values()):
            raise ValueError("baseline requires --width, --height and --timestamp-unit, or a matching --manifest")
        events = load_events(source, **dimensions)
        if manifest and config.end_s is None:
            config = replace(config, end_s=manifest["normalized_interval_end_s"])
        provenance = {"data_kind": "real_davis" if manifest else "user_supplied", "dataset": manifest}
        if args.config:
            provenance["config_file_sha256"] = sha256_file(args.config)
        if args.manifest:
            provenance["manifest_sha256"] = sha256_file(args.manifest)
    provenance.update({"source_path": str(source), "source_sha256": sha256_file(source),
                       "command": ["adaptive-event-vo", *(argv if argv is not None else sys.argv[1:])]})
    summary = run_baseline(events, args.output_dir, config=config, provenance=provenance)
    for method in summary["methods"]:
        print(f"{method['window_ms']:g} ms: {method['processed_events_per_repetition']:,} events, {method['representations_per_repetition']} frames, {method['successful_pairs']}/{method['flow_pairs_per_repetition']} tracked pairs")
    print(f"Image-motion results: {args.output_dir / 'summary.json'}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
