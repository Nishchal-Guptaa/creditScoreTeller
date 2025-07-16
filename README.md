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


## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────┐
│                                 │
│   DeFi Credit Scoring System    │
│                                 │
└───────────────┬─────────────────┘
                │
                v
┌─────────────────────────────────┐
│                                 │
│        Data Collection          │  <─── Blockchain Nodes
│      (Transaction History)      │       DeFi Subgraphs
│                                 │
└───────────────┬─────────────────┘
                │
                v
┌─────────────────────────────────┐
│                                 │
│     Feature Engineering Layer   │
│                                 │
└───────────────┬─────────────────┘
                │
                v
┌─────────────────────────────────┐
│                                 │
│ Hybrid Scoring Engine           │
│ ┌─────────────────────────────┐ │
│ │   Heuristic Rule-Based      │ │
│ │         Scoring             │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │   Machine Learning          │ │
│ │    (Random Forest)          │ │
│ └─────────────────────────────┘ │
│                                 │
└───────────────┬─────────────────┘
                │
                v
┌─────────────────────────────────┐
│                                 │
│    Credit Assessment API       │
│                                 │
└───────────────┬─────────────────┘
                │
                v
┌─────────────────────────────────┐
│                                 │
│      Risk Categorization        │
│     Dashboard & Reporting       │
│                                 │
└─────────────────────────────────┘
```

## Detailed Processing Flow

1. **Data Ingestion Phase**:
   - Transaction data is collected from JSON files containing wallet activity
   - Data includes: wallet addresses, timestamps, actions (deposit/redeem), asset details
   - Input validation and cleaning performed during ingestion

2. **Feature Engineering Phase**:
   - Transaction metrics:
     - Total transaction count
     - Deposit/redeem ratios
     - Transaction frequency
   - Temporal features:
     - Days active
     - Average time between transactions
     - Transaction regularity
   - Asset features:
     - Unique asset count
     - Volume metrics (average, total, max)
   - Behavioral features:
     - Large transaction patterns
     - Interaction patterns with protocols

3. **Scoring Phase**:
   - **Heuristic Scoring** (First Pass):
     - Base score of 500 points
     - Adjustments based on 15+ financial behavior rules
     - Score range: 0-1000
   - **Machine Learning Enhancement**:
     - Random Forest trained on heuristic scores
     - Feature importance weighting
     - Non-linear relationships modeling

4. **Risk Categorization**:
   - Excellent (800+)
   - Good (700-799)
   - Fair (600-699)
   - Poor (500-599)
   - Very Poor (<500)

5. **Reporting Phase**:
   - Individual wallet reports
   - Portfolio-level analytics
   - Risk distribution visualization

## Methodology Selection

### Why Hybrid Approach?

1. **Heuristic Rules**:
   - Provide domain-aware baselines
   - Incorporates financial best practices
   - Transparent and explainable
   - Handles cold-start cases

2. **Machine Learning**:
   - Captures complex patterns
   - Learns from actual behavior data
   - Adaptable to new protocols
   - Handles non-linear relationships

### Why Random Forest?

1. **Advantages**:
   - Handles mixed feature types well
   - Robust to outliers
   - Built-in feature importance
   - Good performance with default parameters
   - Parallelizable (n_jobs=-1)

2. **Alternatives Considered**:
   - Logistic Regression: Too linear for this use case
   - XGBoost: More prone to overfitting with limited data
   - Neural Networks: Overkill for feature set size

### Feature Scaling Choice (StandardScaler)
- Important for distance-based algorithms
- Preserves sparse data characteristics
- Handles extreme values gracefully
- More robust than MinMaxScaling with outliers

## Implementation Details

### Key Components

1. `DeFiCreditScorer` Class:
   - Central orchestration
   - Maintains state (scaler, model)
   - Modular scoring steps

2. Feature Engineering Methods:
   - Time-window based aggregations
   - Protocol-specific normalizations
   - Behavioral pattern detection

3. Model Training Approach:
   - 80/20 train-test split
   - Early stopping validation
   - Hyperparameters tuned for bias-variance tradeoff

### Performance Considerations

1. Optimizations:
   - Vectorized feature calculations
   - Parallel model training (n_jobs=-1)
   - Batch processing of wallets

2. Scaling Characteristics:
   - Linear time complexity with wallets
   - Sub-linear with transaction count
   - Memory efficient feature representations

## Usage Examples

### Basic Implementation

```python
# Initialize scorer
scorer = DeFiCreditScorer()

# Load transaction data
tx_data = scorer.load_transactions('transactions.json')

# Generate features
features = scorer.engineer_features(tx_data)

# Get credit scores
scores = scorer.predict_credit_scores(features)

# Generate report
report = scorer.generate_credit_report(features, scores)
```

### Advanced Configuration

```python
# Custom model parameters
custom_model = RandomForestRegressor(
    n_estimators=150,
    max_depth=12,
    min_samples_leaf=5
)

# Initialize with custom model
scorer = DeFiCreditScorer(model=custom_model)
``` 

## Future Enhancements

1. Planned Features:
   - Real-time scoring API
   - Protocol-specific model variants
   - On-chain reputation integration

2. Research Directions:
   - Graph-based relationship modeling
   - Flash loan attack detection
   - DAO governance participation scoring
