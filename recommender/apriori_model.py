"""Apriori model training using apyori and mlxtend libraries."""

import pandas as pd
from apyori import apriori as apyori_apriori
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder


def train_apyori_model(transactions, min_support=0.003, min_confidence=0.2,
                        min_lift=3, min_length=2, max_length=2):
    """Train an Apriori model using the apyori library.

    Args:
        transactions: List of transactions (lists of item strings).
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.
        min_length: Minimum number of items in a rule.
        max_length: Maximum number of items in a rule.

    Returns:
        A list of apyori RelationRecord results.
    """
    rules = apyori_apriori(
        transactions=transactions,
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        min_length=min_length,
        max_length=max_length,
    )
    return list(rules)


def inspect_results(results):
    """Convert raw apyori results into a structured DataFrame.

    Args:
        results: List of apyori RelationRecord objects.

    Returns:
        A pandas DataFrame with columns:
        Left Hand Side, Right Hand Side, Support, Confidence, Lift.
    """
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]

    return pd.DataFrame(
        list(zip(lhs, rhs, supports, confidences, lifts)),
        columns=["Left Hand Side", "Right Hand Side", "Support", "Confidence", "Lift"],
    )


def train_mlxtend_model(transactions, min_support=0.5):
    """Train an Apriori model using the mlxtend library.

    This approach uses TransactionEncoder for one-hot encoding and
    generates both frequent itemsets and association rules.

    Args:
        transactions: List of transactions (lists of item strings).
        min_support: Minimum support threshold for frequent itemsets.

    Returns:
        A tuple of (frequent_itemsets_df, rules_df, market_items_set).
    """
    te = TransactionEncoder()
    te = te.fit(transactions)
    te_array = te.transform(transactions)
    df = pd.DataFrame(te_array, columns=te.columns_)

    market = set(te.columns_)

    frequent_itemsets = mlxtend_apriori(df, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)

    return frequent_itemsets, rules, market
