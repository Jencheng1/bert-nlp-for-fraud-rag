import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random

def generate_synthetic_fraud_data(num_records=1000):
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate base data
    data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(num_records)],
        'timestamp': [(datetime.now() - timedelta(days=random.randint(0, 30))).isoformat() 
                     for _ in range(num_records)],
        'amount': np.random.normal(1000, 500, num_records).round(2),
        'merchant_category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'utilities'], 
                                           num_records),
        'customer_id': [f'CUST{i:04d}' for i in range(num_records)],
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], num_records),
        'location': [f"{random.uniform(-90, 90):.4f},{random.uniform(-180, 180):.4f}" 
                    for _ in range(num_records)],
        'ip_address': [f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}" 
                      for _ in range(num_records)],
        'is_fraud': np.zeros(num_records, dtype=int)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate fraud patterns
    fraud_patterns = [
        # High amount transactions
        (df['amount'] > 5000) & (df['merchant_category'] == 'travel'),
        # Multiple transactions in short time
        (df['amount'] > 2000) & (df['device_type'] == 'mobile'),
        # Unusual locations
        (df['amount'] > 3000) & (df['merchant_category'] == 'retail'),
        # High-risk merchant categories
        (df['amount'] > 4000) & (df['merchant_category'] == 'entertainment')
    ]
    
    # Apply fraud patterns with some randomness
    for pattern in fraud_patterns:
        fraud_indices = pattern[pattern].index
        num_fraud = len(fraud_indices)
        selected_fraud = np.random.choice(fraud_indices, size=int(num_fraud * 0.3), replace=False)
        df.loc[selected_fraud, 'is_fraud'] = 1
    
    # Add some random noise to fraud labels
    noise_indices = np.random.choice(df.index, size=int(num_records * 0.05), replace=False)
    df.loc[noise_indices, 'is_fraud'] = 1 - df.loc[noise_indices, 'is_fraud']
    
    # Generate transaction descriptions
    df['description'] = df.apply(lambda row: generate_transaction_description(row), axis=1)
    
    return df

def generate_transaction_description(row):
    base_descriptions = {
        'retail': [
            f"Purchase at {random.choice(['Walmart', 'Target', 'Amazon', 'Best Buy'])}",
            f"Online shopping at {random.choice(['eBay', 'Etsy', 'Shopify'])}",
            f"Retail transaction at {random.choice(['Costco', 'Sam\'s Club', 'Home Depot'])}"
        ],
        'food': [
            f"Restaurant payment at {random.choice(['McDonald\'s', 'Starbucks', 'Chipotle'])}",
            f"Food delivery from {random.choice(['UberEats', 'DoorDash', 'Grubhub'])}",
            f"Grocery shopping at {random.choice(['Whole Foods', 'Trader Joe\'s', 'Kroger'])}"
        ],
        'travel': [
            f"Airline ticket booking with {random.choice(['Delta', 'United', 'American Airlines'])}",
            f"Hotel reservation at {random.choice(['Marriott', 'Hilton', 'Hyatt'])}",
            f"Car rental from {random.choice(['Hertz', 'Avis', 'Enterprise'])}"
        ],
        'entertainment': [
            f"Streaming service subscription {random.choice(['Netflix', 'Hulu', 'Disney+'])}",
            f"Gaming platform purchase {random.choice(['Steam', 'PlayStation', 'Xbox'])}",
            f"Event ticket purchase {random.choice(['Ticketmaster', 'Eventbrite', 'StubHub'])}"
        ],
        'utilities': [
            f"Utility bill payment for {random.choice(['Electricity', 'Water', 'Gas'])}",
            f"Internet service payment to {random.choice(['Comcast', 'Verizon', 'AT&T'])}",
            f"Phone bill payment to {random.choice(['Verizon', 'AT&T', 'T-Mobile'])}"
        ]
    }
    
    category = row['merchant_category']
    amount = row['amount']
    description = random.choice(base_descriptions[category])
    
    if row['is_fraud']:
        suspicious_patterns = [
            f"URGENT: {description}",
            f"IMPORTANT: {description}",
            f"PRIORITY: {description}",
            f"VERIFY: {description}",
            f"CONFIRM: {description}"
        ]
        description = random.choice(suspicious_patterns)
    
    return f"{description} - Amount: ${amount:.2f}"

def save_data(df, output_file='data/sample_fraud_data.json'):
    # Convert DataFrame to list of dictionaries
    records = df.to_dict('records')
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Generated {len(records)} records with {df['is_fraud'].sum()} fraudulent transactions")
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_fraud_data(num_records=1000)
    
    # Save to JSON file
    save_data(df) 