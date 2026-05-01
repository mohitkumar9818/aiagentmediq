"""
Launcher so `python app.py` works from the `backend` directory.
"""

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    backend_dir = Path(__file__).resolve().parent
    api_dir = backend_dir / "api"
    os.chdir(api_dir)
    return subprocess.call([sys.executable, "app.py"])


if __name__ == "__main__":
    raise SystemExit(main())
