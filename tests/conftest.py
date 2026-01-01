import sys
from pathlib import Path

HERE = Path(__file__).resolve()

# Look for a parent directory that contains src/api_app.py
for parent in HERE.parents:
    candidate = parent / "src" / "api_app.py"
    if candidate.exists():
        sys.path.insert(0, str(parent))
        break
else:
    raise RuntimeError(
        f"Could not find 'src/api_app.py' above {HERE}. "
        f"Make sure your repo has a 'src' folder with api_app.py."
    )
