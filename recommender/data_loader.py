"""Data loading and preprocessing utilities for the Apriori recommender."""

import os

import pandas as pd


def load_transactions_from_csv(filepath, num_rows=None):
    """Load a CSV file and convert it into a list of transactions.

    Each row in the CSV becomes a transaction (list of item strings).
    NaN values are filtered out.

    Args:
        filepath: Path to the CSV file (no header expected).
        num_rows: Number of rows to load. If None, loads all rows.

    Returns:
        A list of transactions, where each transaction is a list of item strings.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    dataset = pd.read_csv(filepath, header=None)

    if num_rows is not None:
        dataset = dataset.head(num_rows)

    transactions = []
    for i in range(len(dataset)):
        transaction = [
            str(dataset.values[i, j])
            for j in range(dataset.shape[1])
            if str(dataset.values[i, j]) != "nan"
        ]
        transactions.append(transaction)

    return transactions


def load_transactions_from_list(dataset):
    """Wrap a list-of-lists dataset for use with the mlxtend pipeline.

    Args:
        dataset: A list of lists, where each inner list is a transaction
                 of item strings.

    Returns:
        The dataset unchanged (pass-through for API consistency).
    """
    return dataset


def get_unique_items(transactions):
    """Extract the set of unique items from a list of transactions.

    Args:
        transactions: A list of transactions (lists of item strings).

    Returns:
        A set of unique item strings.
    """
    items = set()
    for transaction in transactions:
        items.update(transaction)
    items.discard("nan")
    return items
