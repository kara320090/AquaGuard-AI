from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_APP = ROOT / "app.py"


def main() -> None:
    """Deployment wrapper: run the repository-root Streamlit dashboard."""
    spec = importlib.util.spec_from_file_location("aquaguard_dashboard_app", ROOT_APP)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Streamlit entry file을 찾을 수 없습니다: {ROOT_APP}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
