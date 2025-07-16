# DeFi Credit Scorer

## Overview
The DeFi Credit Scorer is a Python-based application designed to analyze transaction data from decentralized finance (DeFi) platforms. It calculates credit scores for user wallets based on their transaction history, employing both heuristic rules and machine learning techniques. The application provides a detailed credit report, including risk categories and summary statistics.

## Features
- Load transaction data from a JSON file.
- Engineer features from transaction data, including transaction counts, asset diversity, and behavioral patterns.
- Calculate base credit scores using heuristic rules.
- Train a Random Forest regression model to predict credit scores.
- Generate a detailed credit report with risk categories and summary statistics.
- Save the credit report to a CSV file.

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `json`
  
You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

## Usage
1. **Load Transaction Data**: The application expects a JSON file containing transaction data. The JSON structure should include fields such as `userWallet`, `network`, `protocol`, `timestamp`, `action`, and `actionData`.

2. **Run the Application**: You can run the application by executing the script. Make sure to provide the path to your JSON file in the `main` function.

```python
if __name__ == "__main__":
    report = main("path/to/your/transactions.json")
```

3. **Sample Data**: For demonstration purposes, a sample JSON file is created within the script. You can modify the path to your actual transaction data.

## Functions
- `load_transactions(json_file_path)`: Loads transaction data from a specified JSON file.
- `engineer_features(df)`: Creates features from the transaction data for credit scoring.
- `calculate_base_credit_score(features_df)`: Computes base credit scores using heuristic rules.
- `prepare_training_data(features_df)`: Prepares the feature set for training the machine learning model.
- `train_model(features_df, target_scores)`: Trains the Random Forest model on the prepared data.
- `predict_credit_scores(features_df)`: Predicts credit scores for new data using the trained model.
- `generate_credit_report(features_df, scores)`: Generates a detailed credit report including risk categories.

## Output
The application generates a credit score report that includes:
- User wallet addresses
- Credit scores
- Risk categories (Excellent, Good, Fair, Poor, Very Poor)
- Transaction metrics such as total transactions, days active, total volume in USD, unique assets, and transaction frequency.

The report is saved as a CSV file named `credit_score_report.csv`.

## Example
To run the application with sample data, ensure the sample data is saved in a JSON file and modify the path in the `main` function accordingly. The application will output the credit score report to the console and save it to a CSV file.
