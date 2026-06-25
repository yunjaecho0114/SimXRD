"""SimXRD package metadata and runtime setup.

SimXRD implements physically based simulation of powder X-ray diffraction
patterns.

Bin Cao, PhD of HKUST(Guangzhou), https://bin-cao.github.io
URL : https://github.com/Bin-Cao/PyWPEM
"""

from __future__ import annotations

import datetime
import os
import tempfile
from typing import Optional

__title__ = "Pysimxrd"
__description__ = "Physical Simulation of Powder X-ray Diffraction Patterns"
__author__ = "Cao Bin"
__affiliation__ = "Hong Kong University of Science and Technology (GuangZhou)"
__email__ = "bcao686@connect.hkust-gz.edu.cn"
__url__ = "https://bin-cao.github.io"


def _ensure_writable_dir(path: str) -> str:
    """Create and return a writable directory path.

    Args:
        path (str): Directory path to create if it does not already exist.

    Returns:
        str: The same path, after ensuring the directory exists.
    """
    os.makedirs(path, exist_ok=True)
    return path


def configure_runtime_cache() -> None:
    """Set writable cache locations before plotting/font libraries are imported.

    Matplotlib and fontconfig may otherwise try to write under a user-level
    cache directory that is unavailable in sandboxed or batch environments.
    """
    cache_root = _ensure_writable_dir(os.path.join(tempfile.gettempdir(), "simxrd-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", cache_root)
    os.environ.setdefault("MPLCONFIGDIR", _ensure_writable_dir(os.path.join(cache_root, "matplotlib")))


def package_info(executed_at: Optional[datetime.datetime] = None) -> dict[str, str]:
    """Return package metadata for logs or CLI output.

    Args:
        executed_at: Optional timestamp to include in the metadata. If omitted,
            the current local time is used.

    Returns:
        dict[str, str]: Stable package metadata fields.
    """
    executed_at = executed_at or datetime.datetime.now()
    return {
        "package": __title__,
        "description": __description__,
        "author": __author__,
        "affiliation": __affiliation__,
        "email": __email__,
        "url": __url__,
        "executed_at": executed_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def format_package_info(executed_at: Optional[datetime.datetime] = None) -> str:
    """Format package metadata as a short banner.

    Args:
        executed_at: Optional timestamp to include in the banner.

    Returns:
        str: Multi-line banner text suitable for interactive runs.
    """
    info = package_info(executed_at)
    line = "=" * 80
    return "\n".join([
        line,
        f"{info['package']}: {info['description']}",
        f"Author: {info['author']}, HKUST(GZ) | {info['url']}",
        f"Email: {info['email']}",
        f"Executed on: {info['executed_at']}",
        line,
    ])


def print_package_info(executed_at: Optional[datetime.datetime] = None) -> None:
    """Print package metadata for interactive runs.

    Args:
        executed_at: Optional timestamp to include in the printed banner.
    """
    print(format_package_info(executed_at))


configure_runtime_cache()

__all__ = [
    "__title__",
    "__description__",
    "__author__",
    "__affiliation__",
    "__email__",
    "__url__",
    "configure_runtime_cache",
    "package_info",
    "format_package_info",
    "print_package_info",
]
