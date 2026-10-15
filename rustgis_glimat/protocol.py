"""The plugin's side of RustGIS's process-plugin protocol: JSON lines on standard output.

Imported first by ``main.py``, which then points ``sys.stdout`` at a :class:`Progress` writer:
the protocol keeps the real standard output to itself, so what GLIMAT and its libraries print
can't corrupt it, and what they print shows in RustGIS as the tool's progress.
"""

import json
import math
import sys
import threading

_PROTOCOL = sys.stdout
# GLIMAT's parallel analyses print from several threads.
_LOCK = threading.Lock()


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
    with _LOCK:
        _PROTOCOL.write(text + "\n")
        _PROTOCOL.flush()


def log(text):
    """Adds a line to the tool run's messages in RustGIS.

    Args:
        text: The line.
    """
    send({"log": str(text)})


def progress(text):
    """Shows what the tool is doing now in RustGIS, while it runs; it isn't kept.

    Args:
        text: The line.
    """
    send({"progress": str(text)})


class Progress:
    """Stands in for ``sys.stdout``: each line printed goes to standard error, which RustGIS
    logs, and is shown as the tool's progress.

    Lines with no letters or digits (GLIMAT's ``====`` rules, blank lines) aren't shown.
    """

    def __init__(self):
        """Starts with no partial line."""
        self._partial = ""
        self._lock = threading.Lock()
        self.encoding = sys.stderr.encoding

    def write(self, text):
        """Writes ``text``, reporting each line it completes.

        Args:
            text: What was printed.

        Returns:
            The number of characters written.
        """
        sys.stderr.write(text)
        with self._lock:
            *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            # A progress bar redraws its line after a carriage return: the last drawing counts.
            line = line.rsplit("\r", 1)[-1].strip()
            if any(c.isalnum() for c in line):
                progress(line)
        return len(text)

    def flush(self):
        """Flushes standard error."""
        sys.stderr.flush()

    def isatty(self):
        """Says it's not a terminal, so libraries don't draw for one.

        Returns:
            False.
        """
        return False
