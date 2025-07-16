import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class DeFiCreditScorer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.feature_columns = []
        
    def load_transactions(self, json_file_path):
        """Load transactions from JSON file"""
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return None
    
    def engineer_features(self, df):
        """Engineer features from transaction data"""
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Group by wallet for feature engineering
        wallet_features = []
        
        for wallet in df['userWallet'].unique():
            wallet_txs = df[df['userWallet'] == wallet].copy()
            wallet_txs = wallet_txs.sort_values('datetime')
            
            features = {
                'userWallet': wallet,
                'network': wallet_txs['network'].iloc[0],
                'protocol': wallet_txs['protocol'].iloc[0]
            }
            
            # Basic transaction metrics
            features['total_transactions'] = len(wallet_txs)
            features['unique_actions'] = wallet_txs['action'].nunique()
            features['deposit_count'] = len(wallet_txs[wallet_txs['action'] == 'deposit'])
            features['redeem_count'] = len(wallet_txs[wallet_txs['action'] == 'redeemunderlying'])
            
            # Time-based features
            features['days_active'] = (wallet_txs['datetime'].max() - wallet_txs['datetime'].min()).days + 1
            features['avg_time_between_txs'] = np.mean(np.diff(wallet_txs['datetime'].astype(int) // 10**9)) if len(wallet_txs) > 1 else 0
            
            # Asset diversity
            features['unique_assets'] = wallet_txs['actionData'].apply(
                lambda x: x.get('assetSymbol', 'UNKNOWN')
            ).nunique()
            
            # Transaction amounts and values
            amounts = []
            usd_values = []
            
            for _, tx in wallet_txs.iterrows():
                action_data = tx['actionData']
                amount = float(action_data.get('amount', 0))
                price = float(action_data.get('assetPriceUSD', 0))
                
                amounts.append(amount)
                usd_values.append(amount * price / 10**18)  # Assuming 18 decimals for most tokens
            
            features['avg_transaction_amount'] = np.mean(amounts) if amounts else 0
            features['total_volume_usd'] = sum(usd_values)
            features['avg_transaction_value_usd'] = np.mean(usd_values) if usd_values else 0
            features['max_transaction_value_usd'] = max(usd_values) if usd_values else 0
            features['min_transaction_value_usd'] = min(usd_values) if usd_values else 0
            features['transaction_value_std'] = np.std(usd_values) if len(usd_values) > 1 else 0
            
            # Behavioral patterns
            features['deposit_to_total_ratio'] = features['deposit_count'] / features['total_transactions']
            features['redeem_to_total_ratio'] = features['redeem_count'] / features['total_transactions']
            
            # Regularity metrics
            if len(wallet_txs) > 1:
                time_diffs = np.diff(wallet_txs['datetime'].astype(int) // 10**9)
                features['time_regularity'] = 1 / (1 + np.std(time_diffs))
                features['max_time_gap'] = max(time_diffs)
            else:
                features['time_regularity'] = 0
                features['max_time_gap'] = 0
            
            # Risk indicators
            features['large_transaction_ratio'] = sum(1 for v in usd_values if v > np.mean(usd_values) * 3) / len(usd_values) if usd_values else 0
            features['transaction_frequency'] = features['total_transactions'] / max(features['days_active'], 1)
            
            # Asset price volatility awareness
            price_changes = []
            for _, tx in wallet_txs.iterrows():
                price_changes.append(float(tx['actionData'].get('assetPriceUSD', 0)))
            
            features['price_volatility_exposure'] = np.std(price_changes) if len(price_changes) > 1 else 0
            
            wallet_features.append(features)
        
        return pd.DataFrame(wallet_features)
    
    def calculate_base_credit_score(self, features_df):
        """Calculate base credit scores using heuristic rules"""
        scores = []
        
        for _, row in features_df.iterrows():
            score = 500  # Base score
            
            # Transaction volume and consistency (0-200 points)
            if row['total_transactions'] >= 10:
                score += 50
            elif row['total_transactions'] >= 5:
                score += 25
            
            if row['days_active'] >= 30:
                score += 30
            elif row['days_active'] >= 7:
                score += 15
            
            # Deposit/withdrawal balance (0-100 points)
            balance_ratio = abs(row['deposit_to_total_ratio'] - 0.5)
            score += int(50 * (1 - balance_ratio * 2))
            
            # Asset diversity (0-100 points)
            if row['unique_assets'] >= 3:
                score += 50
            elif row['unique_assets'] >= 2:
                score += 25
            
            # Transaction regularity (0-100 points)
            score += int(row['time_regularity'] * 50)
            
            # Volume consistency (0-100 points)
            if row['total_volume_usd'] > 1000:
                score += 40
            elif row['total_volume_usd'] > 100:
                score += 20
            
            # Penalize suspicious patterns
            if row['transaction_frequency'] > 10:  # Too frequent
                score -= 100
            
            if row['large_transaction_ratio'] > 0.8:  # Too many large transactions
                score -= 50
            
            if row['max_time_gap'] > 86400 * 90:  # Long gaps (>90 days)
                score -= 30
            
            # Ensure score is within bounds
            score = max(0, min(1000, score))
            scores.append(score)
        
        return np.array(scores)
    
    def prepare_training_data(self, features_df):
        """Prepare features for ML model training"""
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-related columns if they exist
        exclude_cols = ['userWallet', 'network', 'protocol']
        numeric_features = [col for col in numeric_features if col not in exclude_cols]
        
        self.feature_columns = numeric_features
        X = features_df[numeric_features].fillna(0)
        
        return X
    
    def train_model(self, features_df, target_scores):
        """Train the machine learning model"""
        X = self.prepare_training_data(features_df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, target_scores, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        
        return self.model
    
    def predict_credit_scores(self, features_df):
        """Predict credit scores for new data"""
        X = self.prepare_training_data(features_df)
        X_scaled = self.scaler.transform(X)
        
        scores = self.model.predict(X_scaled)
        
        # Ensure scores are within bounds
        scores = np.clip(scores, 0, 1000)
        
        return scores
    
    def generate_credit_report(self, features_df, scores):
        """Generate detailed credit report"""
        report = features_df.copy()
        report['credit_score'] = scores
        
        # Add risk categories
        def risk_category(score):
            if score >= 800:
                return 'Excellent'
            elif score >= 700:
                return 'Good'
            elif score >= 600:
                return 'Fair'
            elif score >= 500:
                return 'Poor'
            else:
                return 'Very Poor'
        
        report['risk_category'] = report['credit_score'].apply(risk_category)
        
        # Sort by score
        report = report.sort_values('credit_score', ascending=False)
        
        return report[['userWallet', 'credit_score', 'risk_category', 'total_transactions', 
                      'days_active', 'total_volume_usd', 'unique_assets', 'transaction_frequency']]

def main(json_file_path):
    """Main function to process JSON file and generate credit scores"""
    
    print("DeFi Credit Scoring System")
    print("=" * 50)
    
    # Initialize scorer
    scorer = DeFiCreditScorer()
    
    # Load transaction data
    print("Loading transaction data...")
    df = scorer.load_transactions(json_file_path)
    if df is None:
        return
    
    print(f"✅ Loaded {len(df)} transactions for {df['userWallet'].nunique()} wallets")
    
    # Engineer features
    print("🔧 Engineering features...")
    features_df = scorer.engineer_features(df)
    print(f"✅ Generated {len(features_df.columns)} features")
    
    # Calculate base scores using heuristics
    print("🎯 Calculating base credit scores...")
    base_scores = scorer.calculate_base_credit_score(features_df)
    
    # Train ML model
    print("🤖 Training machine learning model...")
    scorer.train_model(features_df, base_scores)
    
    # Generate final scores
    print("📈 Generating final credit scores...")
    final_scores = scorer.predict_credit_scores(features_df)
    
    # Generate report
    print("📋 Generating credit report...")
    report = scorer.generate_credit_report(features_df, final_scores)
    
    # Display results
    print("\n" + "=" * 80)
    print("🏆 CREDIT SCORE REPORT")
    print("=" * 80)
    print(report.to_string(index=False))
    
    # Summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Average Credit Score: {final_scores.mean():.2f}")
    print(f"Median Credit Score: {np.median(final_scores):.2f}")
    print(f"Standard Deviation: {final_scores.std():.2f}")
    print(f"Highest Score: {final_scores.max():.2f}")
    print(f"Lowest Score: {final_scores.min():.2f}")
    
    # Risk distribution
    risk_dist = report['risk_category'].value_counts()
    print(f"\n🎯 Risk Distribution:")
    for category, count in risk_dist.items():
        print(f"{category}: {count} wallets ({count/len(report)*100:.1f}%)")
        
    report.to_csv('credit_score_report.csv', index=False)
    
    return "Credit report saved to 'credit_score_report.csv'"

# Example usage with sample data
if __name__ == "__main__":
    # If you have a JSON file, uncomment the line below and provide the path
    # report = main("path/to/your/transactions.json")
    
    # For demonstration with the provided sample data
    sample_data = [
        {
            "_id": {"$oid": "681d38fed63812d4655f571a"},
            "userWallet": "0x00000000001accfa9cef68cf5371a23025b6d4b6",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": "0x695c69acf608fbf5d38e48ca5535e118cc213a89e3d6d2e66e6b0e3b2e8d4190",
            "timestamp": 1629178166,
            "action": "deposit",
            "actionData": {
                "type": "Deposit",
                "amount": "2000000000",
                "assetSymbol": "USDC",
                "assetPriceUSD": "0.9938318274296357543568636362026045",
                "poolId": "0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
                "userId": "0x00000000001accfa9cef68cf5371a23025b6d4b6"
            }
        },
        {
            "_id": {"$oid": "681aa70dd6df53021cc6f3c0"},
            "userWallet": "0x000000000051d07a4fb3bd10121a343d85818da6",
            "network": "polygon",
            "protocol": "aave_v2",
            "txHash": "0xe6fc162c86b2928b0ba9b82bda672763665152b9de9d92b0e1512a81b1129e3f",
            "timestamp": 1621525013,
            "action": "deposit",
            "actionData": {
                "type": "Deposit",
                "amount": "145000000000000000000",
                "assetSymbol": "WMATIC",
                "assetPriceUSD": "1.970306761113742502077627085754506",
                "poolId": "0x0d500b1d8e8ef31e21c99d1db9a6444d3adf1270",
                "userId": "0x000000000051d07a4fb3bd10121a343d85818da6"
            }
        }
    ]
    
    # Save sample data to JSON file for demonstration
    with open('sample_transactions.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("🚀 Running with actual data...")
    report = main('path/to/your/transactions.json')
