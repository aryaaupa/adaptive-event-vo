"""Input and implementation fingerprints for local research artifacts."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import os
import platform
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from . import __version__


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0" + path.read_bytes())
    cpu = platform.processor()
    if not cpu and Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown")
    try:
        distribution_version = version("opencv-python-headless")
    except PackageNotFoundError:
        distribution_version = None
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count(), "numpy": np.__version__, "opencv": cv2.__version__,
            "opencv_python_headless": distribution_version, "matplotlib": matplotlib.__version__,
            "opencv_threads": cv2.getNumThreads(), "opencl_enabled": bool(cv2.ocl.useOpenCL()),
            "adaptive_event_vo": __version__, "implementation_sha256": digest.hexdigest()}
