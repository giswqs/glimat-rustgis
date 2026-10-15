"""Builds the plugin package, ``dist/glimat-<version>.rgplugin``, for installing in RustGIS.

A package is a zip archive with ``plugin.json`` at its root (see RustGIS's ``docs/plugins.md``).
The archive is reproducible: the same files give the same bytes, and so the same SHA-256.

Usage: ``python scripts/package.py [--out DIR]``. Prints the package's path and SHA-256.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What the plugin needs to run, and its license and documentation. Tests aren't packaged.
FILES = ["plugin.json", "main.py", "requirements.txt", "LICENSE", "README.md"]
FOLDERS = ["glimat", "rustgis_glimat"]

# A fixed time for every entry, so the archive doesn't change with the files' times.
TIME = (2026, 1, 1, 0, 0, 0)


def members():
    """Lists the files to package, relative to the repository's root.

    Returns:
        The paths, sorted, so the archive's order is stable.

    Raises:
        FileNotFoundError: When a file the plugin needs is missing.
    """
    paths = [ROOT / f for f in FILES]
    for folder in FOLDERS:
        paths += (ROOT / folder).glob("*.py")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    return sorted(p.relative_to(ROOT).as_posix() for p in paths)


def build(out):
    """Writes the package to ``out``.

    Args:
        out: The folder to write the package to; made when it doesn't exist.

    Returns:
        The package's path.
    """
    manifest = json.loads((ROOT / "plugin.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{manifest['id']}-{manifest['version']}.rgplugin"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in members():
            info = zipfile.ZipInfo(name, TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, (ROOT / name).read_bytes())
    return path


def main():
    """Builds the package and prints its path and SHA-256."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", type=Path, default=ROOT / "dist", help="output folder (dist/)"
    )
    path = build(parser.parse_args().out)
    print(path)
    print(hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
