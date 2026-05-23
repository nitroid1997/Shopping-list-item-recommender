"""Visualization utilities for Apriori association rule results."""

import pandas as pd


def display_top_rules(results_df, n=10, sort_by="Lift"):
    """Display the top N association rules sorted by a given metric.

    Args:
        results_df: DataFrame from inspect_results() with columns
                    Left Hand Side, Right Hand Side, Support, Confidence, Lift.
        n: Number of top rules to display.
        sort_by: Column name to sort by (descending).

    Returns:
        A DataFrame containing the top N rules.
    """
    top_rules = results_df.nlargest(n=n, columns=sort_by)
    return top_rules


def print_rules_table(results_df, n=10, sort_by="Lift"):
    """Print a formatted table of the top association rules.

    Args:
        results_df: DataFrame with association rule results.
        n: Number of rules to display.
        sort_by: Column to sort by.
    """
    top = display_top_rules(results_df, n=n, sort_by=sort_by)

    print(f"\n{'='*80}")
    print(f"  Top {n} Association Rules (sorted by {sort_by})")
    print(f"{'='*80}")
    print(f"{'LHS':<25} {'RHS':<25} {'Support':<10} {'Confidence':<12} {'Lift':<8}")
    print(f"{'-'*80}")

    for _, row in top.iterrows():
        print(
            f"{row['Left Hand Side']:<25} "
            f"{row['Right Hand Side']:<25} "
            f"{row['Support']:<10.4f} "
            f"{row['Confidence']:<12.4f} "
            f"{row['Lift']:<8.4f}"
        )

    print(f"{'='*80}\n")


def print_summary(results_df):
    """Print a summary of the association rules analysis.

    Args:
        results_df: DataFrame with association rule results.
    """
    print(f"\n--- Association Rules Summary ---")
    print(f"Total rules discovered: {len(results_df)}")
    print(f"Average support:    {results_df['Support'].mean():.4f}")
    print(f"Average confidence: {results_df['Confidence'].mean():.4f}")
    print(f"Average lift:       {results_df['Lift'].mean():.4f}")
    print(f"Max lift:           {results_df['Lift'].max():.4f}")
    print()
