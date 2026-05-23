# Shopping List Item Recommender

A shopping list item recommendation system based on the **Apriori algorithm** for market basket analysis. Given historical transaction data, the system discovers association rules between items and provides real-time purchase suggestions as you build your cart.

## Project Structure

```
Shopping-list-item-recommender/
├── main.py                     # Application entry point
├── recommender/                # Core package
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── apriori_model.py        # Apriori model training (apyori + mlxtend)
│   ├── visualization.py        # Results display and formatting
│   └── predictor.py            # Interactive cart prediction system
├── data/
│   └── Market_Basket_Optimisation.csv   # Sample transaction dataset
├── notebooks/
│   └── Apriori_item_predictor.ipynb     # Original Jupyter notebook
├── requirements.txt            # Python dependencies
├── setup.sh                    # Dependency check and installation script
└── .gitignore
```

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/nitroid1997/Shopping-list-item-recommender.git
cd Shopping-list-item-recommender

# Run the setup script (creates venv, installs dependencies, verifies imports)
chmod +x setup.sh
./setup.sh

# Activate the virtual environment
source venv/bin/activate
```

### 2. Run

```bash
# Full pipeline: market basket analysis + interactive cart
python main.py

# Market basket analysis only (discovers and displays association rules)
python main.py --analyze

# Interactive cart prediction only (uses sample data)
python main.py --predict

# Use a custom dataset
python main.py --data path/to/your/data.csv

# Customize analysis parameters
python main.py --top-n 20 --min-support 0.3
```

## Features

### Market Basket Analysis (`--analyze`)
- Loads transaction data from CSV files
- Trains an Apriori model using the **apyori** library
- Discovers association rules with configurable support, confidence, and lift thresholds
- Displays the top rules in a formatted table with summary statistics

### Interactive Cart Prediction (`--predict`)
- Trains an Apriori model using **mlxtend** for real-time predictions
- Interactive CLI where you add items to your cart
- Suggests related items based on learned association rules
- Tracks remaining available items and current cart contents

## How It Works

The **Apriori algorithm** identifies frequent itemsets in transaction data and generates association rules of the form:

```
{item_A} → {item_B}    (support, confidence, lift)
```

- **Support**: How frequently the itemset appears in the dataset
- **Confidence**: How often the rule is found to be true
- **Lift**: How much more often the items appear together than expected if they were independent

## Dependencies

| Package      | Purpose                              |
|-------------|--------------------------------------|
| numpy       | Numerical computing                  |
| pandas      | Data manipulation and analysis       |
| matplotlib  | Plotting (available for extensions)  |
| apyori      | Apriori algorithm implementation     |
| mlxtend     | Frequent pattern mining and rules    |
| scikit-learn| ML utilities (mlxtend dependency)    |

## Dataset

The included `Market_Basket_Optimisation.csv` contains 7,501 transactions from a grocery store, each with up to 20 items. This is a standard market basket analysis dataset.

## Command-Line Options

| Option          | Default                                  | Description                                |
|----------------|------------------------------------------|--------------------------------------------|
| `--analyze`    | —                                        | Run market basket analysis only            |
| `--predict`    | —                                        | Run interactive cart prediction only       |
| `--data`       | `data/Market_Basket_Optimisation.csv`    | Path to the CSV dataset                    |
| `--top-n`      | `10`                                     | Number of top rules to display             |
| `--min-support`| `0.5`                                    | Minimum support for the prediction model   |

## License

This project is for educational purposes.
