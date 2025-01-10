import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

N_SAMPLES=2000

def generate_synthetic_data(n_samples=N_SAMPLES, n_features=7, n_classes=5, 
                          val_size=0.2, test_size=0.2, random_state=42):
    """Generate synthetic data with realistic class overlap."""
    np.random.seed(random_state)
    
    # Generate base features with moderate scale
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear relationships with noise
    noise_level = 0.3  # Add more noise
    X[:, 0] = np.sin(X[:, 0]) + np.random.randn(n_samples) * noise_level
    X[:, 1] = np.exp(X[:, 1] / 4) + np.random.randn(n_samples) * noise_level
    X[:, 2] = (X[:, 0] * X[:, 1]) / 2 + np.random.randn(n_samples) * noise_level
    X[:, 3] = np.square(X[:, 3]) / 2 + np.random.randn(n_samples) * noise_level
    X[:, 4] = np.tanh(X[:, 4]) + np.random.randn(n_samples) * noise_level
    X[:, 5] = np.cos(X[:, 5]) + np.random.randn(n_samples) * noise_level
    X[:, 6] = np.sign(X[:, 6]) * np.log(np.abs(X[:, 6]) + 1) + np.random.randn(n_samples) * noise_level
    
    # Generate target classes with overlapping boundaries
    logits = np.zeros((n_samples, n_classes))
    logits[:, 0] = np.sin(X[:, 0]) + np.cos(X[:, 1]) / 2 + X[:, 4] / 3
    logits[:, 1] = X[:, 2] * X[:, 3] / 4 - np.square(X[:, 1]) / 2 + X[:, 5] / 3
    logits[:, 2] = np.exp(X[:, 0]/4) - np.sin(X[:, 2] * X[:, 3]) / 2 + X[:, 6] / 3
    logits[:, 3] = np.tanh(X[:, 4] + X[:, 5]) / 2 + np.cos(X[:, 0] * X[:, 1]) / 3
    logits[:, 4] = np.sin(X[:, 6]) / 2 + np.exp(X[:, 2]/4) - np.cos(X[:, 3]) / 3
    
    # Add noise to logits
    logits += np.random.randn(*logits.shape) * 0.2
    
    # Convert to probabilities and sample classes probabilistically
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    y = np.array([np.random.choice(n_classes, p=p) for p in probs])
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create DataFrames
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['target'] = y
    
    # Split data while maintaining approximate class distribution
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=random_state,
        stratify=df['target']
    )
    
    # Split remaining data into train and validation
    adjusted_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val_df['target']
    )
    
    # Print dataset statistics
    print("\nClass distribution:")
    print("\nTraining set:")
    print(train_df['target'].value_counts(normalize=True).sort_index())
    print("\nValidation set:")
    print(val_df['target'].value_counts(normalize=True).sort_index())
    print("\nTest set:")
    print(test_df['target'].value_counts(normalize=True).sort_index())
    
    # Print feature statistics
    print("\nFeature statistics:")
    print(train_df[feature_cols].describe())
    
    return train_df, val_df, test_df

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate data
    train_df, val_df, test_df = generate_synthetic_data(
        n_samples=N_SAMPLES,
        n_features=7,
        n_classes=5,
        val_size=0.2,
        test_size=0.2,
        random_state=42
    )
    
    # Save to CSV files
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print("\nDataset shapes:")
    print(f"Training:   {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test:       {test_df.shape}")
    print("\nData files saved to:")
    print("- data/train.csv")
    print("- data/val.csv")
    print("- data/test.csv")

if __name__ == "__main__":
    main()