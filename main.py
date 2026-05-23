#!/usr/bin/env python3
"""Shopping List Item Recommender — Main entry point.

This application uses the Apriori algorithm to discover association rules
from market basket transaction data, then provides an interactive shopping
cart that suggests items based on those rules.

Usage:
    python main.py                          # Run full pipeline (analysis + interactive cart)
    python main.py --analyze                # Run market basket analysis only
    python main.py --predict                # Run interactive cart with sample data
    python main.py --data path/to/file.csv  # Use a custom dataset for analysis
"""

import argparse
import os
import sys

from recommender.apriori_model import (
    inspect_results,
    train_apyori_model,
    train_mlxtend_model,
)
from recommender.data_loader import load_transactions_from_csv
from recommender.predictor import run_interactive_cart
from recommender.visualization import print_rules_table, print_summary

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Market_Basket_Optimisation.csv")

SAMPLE_TRANSACTIONS = [
    ["Bread", "Milk", "Beer"],
    ["Bread", "Diapers", "Eggs"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Cola"],
]


def run_analysis(data_path, top_n=10):
    """Run market basket analysis on CSV data using apyori.

    Args:
        data_path: Path to the CSV dataset.
        top_n: Number of top rules to display.
    """
    print(f"\nLoading data from: {data_path}")
    transactions = load_transactions_from_csv(data_path)
    print(f"Loaded {len(transactions)} transactions.\n")

    print("Training Apriori model (apyori)...")
    results = train_apyori_model(transactions)
    print(f"Discovered {len(results)} association rules.\n")

    results_df = inspect_results(results)

    print_summary(results_df)
    print_rules_table(results_df, n=top_n, sort_by="Lift")

    return results_df


def run_prediction(transactions=None, min_support=0.5):
    """Run the interactive cart prediction system.

    Args:
        transactions: List of transactions. Uses SAMPLE_TRANSACTIONS if None.
        min_support: Minimum support for the mlxtend model.
    """
    if transactions is None:
        transactions = SAMPLE_TRANSACTIONS
        print("\nUsing sample transaction data for prediction demo.")
    else:
        print(f"\nUsing {len(transactions)} transactions for prediction.")

    print("Training mlxtend Apriori model...")
    _, rules, market = train_mlxtend_model(transactions, min_support=min_support)
    print(f"Generated {len(rules)} association rules.")
    print(f"Available items: {sorted(market)}\n")

    run_interactive_cart(rules, market)


def main():
    parser = argparse.ArgumentParser(
        description="Shopping List Item Recommender using the Apriori algorithm.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run market basket analysis on CSV data.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run the interactive cart prediction system.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV dataset (default: data/Market_Basket_Optimisation.csv).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top rules to display (default: 10).",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.5,
        help="Minimum support for the prediction model (default: 0.5).",
    )

    args = parser.parse_args()

    # Default: run both if neither flag is given
    run_both = not args.analyze and not args.predict

    if args.analyze or run_both:
        if not os.path.isfile(args.data):
            print(f"Error: Dataset not found at '{args.data}'.", file=sys.stderr)
            print("Please provide a valid path with --data.", file=sys.stderr)
            sys.exit(1)
        run_analysis(args.data, top_n=args.top_n)

    if args.predict or run_both:
        run_prediction(min_support=args.min_support)


if __name__ == "__main__":
    main()
