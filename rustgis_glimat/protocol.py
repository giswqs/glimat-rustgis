"""The plugin's side of RustGIS's process-plugin protocol: JSON lines on standard output.

Imported first by ``main.py``, which then points ``sys.stdout`` at standard error: the protocol
keeps the real standard output to itself, so what GLIMAT and its libraries print can't corrupt it.
"""

import json
import math
import sys

_PROTOCOL = sys.stdout


def _json_default(value):
    """Converts what ``json`` can't encode (NumPy numbers and arrays, paths) to what it can.

    Args:
        value: The value ``json`` couldn't encode.

    Returns:
        A JSON-encodable equivalent: a plain number or list, or text.
    """
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _finite(value):
    """Replaces NaN and infinities, which JSON can't carry, with ``None``, at any depth.

    Args:
        value: A JSON-encodable value.

    Returns:
        The value with every non-finite float replaced by ``None``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def send(message):
    """Writes one protocol message to RustGIS.

    Args:
        message: A JSON-encodable dict: ``{"log": ...}``, ``{"ok": ...}`` or ``{"error": ...}``.
    """
    try:
        text = json.dumps(message, default=_json_default, allow_nan=False)
    except ValueError:
        # A NaN somewhere (a metric of an empty time step): sent as null.
        text = json.dumps(
            _finite(json.loads(json.dumps(message, default=_json_default)))
        )
    _PROTOCOL.write(text + "\n")
    _PROTOCOL.flush()


def log(text):
    """Adds a line to the tool run's messages in RustGIS.

    Args:
        text: The line.
    """
    send({"log": str(text)})
