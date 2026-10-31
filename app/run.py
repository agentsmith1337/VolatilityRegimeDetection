"""Start the live runner and the dashboard together (local only). Ctrl+C stops both.

Usage (from the repo root):
    python app/run.py                         # live
    python app/run.py --replay 2025-04-07     # replay a past session (demo)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    args = sys.argv[1:]
    runner = subprocess.Popen([sys.executable, str(ROOT / 'app' / 'runner.py'), *args], cwd=ROOT)
    dash_args = ['--', '--replay'] if '--replay' in args else []
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app' / 'dashboard.py'), *dash_args],
                       cwd=ROOT)
    except KeyboardInterrupt:
        pass
    finally:
        runner.terminate()
