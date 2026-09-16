import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR / "src"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.deo_dash2 import main


if __name__ == "__main__":
    main()
