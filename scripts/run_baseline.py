"""Run fixed-window image-motion baselines using the installed package."""

import sys

from adaptive_event_vo.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["baseline", *sys.argv[1:]]))
