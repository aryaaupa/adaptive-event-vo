"""Generate, load and plot a synthetic stream using the installed package."""

import sys

from adaptive_event_vo.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["demo", *sys.argv[1:]]))
