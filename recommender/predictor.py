"""Interactive shopping cart prediction system using association rules."""


def get_suggestions(item, cart, rules):
    """Get item suggestions based on association rules for a given item.

    Args:
        item: The item just added to the cart (as a string).
        cart: The current cart contents (a set of strings).
        rules: The association rules DataFrame from mlxtend.

    Returns:
        A set of suggested items (excluding items already in the cart).
    """
    item_set = set([item])
    matching = rules[rules["antecedents"].apply(lambda x: item_set.issubset(set(x)))]

    suggestions = set()
    for idx in matching["consequents"].index:
        consequent_items = set(list(matching["consequents"][idx]))
        suggestions |= consequent_items

    suggestions -= cart
    suggestions -= item_set
    return suggestions


def run_interactive_cart(rules, market):
    """Run the interactive cart addition application.

    Users can add items to their cart and receive suggestions based on
    association rules learned from transaction data.

    Args:
        rules: The association rules DataFrame from mlxtend.
        market: A set of all available items in the store.
    """
    available = set(market)
    cart = set()

    print("=" * 70)
    print("  Shopping Cart Item Recommender")
    print("  Type 'exit' to leave")
    print("=" * 70)
    print(f"\nAvailable items: {sorted(available)}\n")

    while True:
        item = input("Enter an item: ").strip()
        if not item:
            continue

        item = item[0].upper() + item[1:].lower()

        if item == "Exit":
            print("\n" + "=" * 70)
            print(f"  Your final cart: {sorted(cart)}")
            print("=" * 70)
            break

        if item in cart:
            print("  Already in cart.\n")
            continue

        if item not in available:
            print("  Item not available.\n")
            continue

        cart.add(item)
        available.discard(item)

        suggestions = get_suggestions(item, cart, rules)

        print("-" * 70)
        print(f"  Cart:        {sorted(cart)}")
        if suggestions:
            print(f"  Suggestions: {sorted(suggestions)}")
        else:
            print("  Suggestions: (none)")
        print(f"  Remaining:   {sorted(available)}")
        print("-" * 70 + "\n")

        if len(available) == 0:
            print("All items taken!")
            print("=" * 70)
            print(f"  Your final cart: {sorted(cart)}")
            print("=" * 70)
            break
