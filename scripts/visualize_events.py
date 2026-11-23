"""Convenience wrapper; install the package first with `pip install -e .`."""

import sys

from adaptive_event_vo.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["view", *sys.argv[1:]]))
