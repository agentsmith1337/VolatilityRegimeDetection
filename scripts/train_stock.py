"""Train and test the regime models on one NSE stock from the command line (same pipeline as the dashboard).

Usage:
    python scripts/train_stock.py RELIANCE
    python scripts/train_stock.py HDFCBANK --models lightgbm xgboost logistic soft_vote --years 8 --test-months 12
    python scripts/train_stock.py RELIANCE --update      # rescore the latest bars with the trained models
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from vrdet.stocks import score_latest, train_stock  # noqa: E402
from vrdet.zoo import MODELS  # noqa: E402

FAST = ['lightgbm', 'xgboost', 'logistic', 'soft_vote']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('symbol', help='NSE symbol as in the Angel One instrument list, e.g. RELIANCE')
    parser.add_argument('--models', nargs='+', default=FAST, choices=list(MODELS))
    parser.add_argument('--years', type=int, default=8, help='years of history (Angel One 15m data starts 2017)')
    parser.add_argument('--test-months', type=int, default=12)
    parser.add_argument('--update', action='store_true', help='only rescore the latest bars (no training)')
    args = parser.parse_args()
    if args.update:
        score_latest(args.symbol.upper())
    else:
        train_stock(args.symbol.upper(), args.models, args.years, args.test_months)
