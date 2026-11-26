"""Download/cache the official DAVIS shapes_rotation archive and convert a clip."""

import argparse
from pathlib import Path

from adaptive_event_vo.datasets.davis import convert_davis_clip, download_shapes_rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/davis/shapes_rotation.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/davis/shapes_rotation_clip"))
    parser.add_argument("--start-s", type=float, default=5)
    parser.add_argument("--duration-s", type=float, default=2)
    args = parser.parse_args()
    archive = download_shapes_rotation(args.archive)
    metadata = convert_davis_clip(archive, args.output_dir, start_s=args.start_s, duration_s=args.duration_s)
    print(f"Prepared {metadata['event_count']:,} real events; manifest: {args.output_dir / 'manifest.json'}")
    print(f"Dataset terms: {metadata['license_url']}; cite {metadata['citation']}")


if __name__ == "__main__":
    main()
