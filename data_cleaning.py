# data_cleaning.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# 1. Import and explore the dataset
def load_and_explore_data():
    # Load the dataset (save it in your project folder first)
    df = pd.read_csv('titanic.csv')  # Make sure to download the dataset first
    
    # Basic exploration
    print("=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    print("\n=== Descriptive Statistics ===")
    print(df.describe())
    
    return df

# 2. Handle missing values
def handle_missing_values(df):
    # Age - fill with median
    df['Age'].fillna(df['Age'].median(), inplace=True)
    
    # Embarked - fill with mode (most frequent value)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Cabin - has too many missing values, we'll drop this column
    df.drop('Cabin', axis=1, inplace=True)
    
    return df

# 3. Convert categorical features to numerical
def encode_categorical_features(df):
    # Label encoding for Sex (binary category)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    
    # One-hot encoding for Embarked (multiple categories)
    embarked_encoded = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_encoded], axis=1)
    df.drop('Embarked', axis=1, inplace=True)
    
    return df

# 4. Normalize/standardize numerical features
def scale_numerical_features(df):
    # Select numerical columns to scale
    numerical_cols = ['Age', 'Fare']
    
    # Standardization (mean=0, std=1)
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

# 5. Visualize and handle outliers
def handle_outliers(df):
    # Visualize outliers using boxplots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['Age', 'Fare']])
    plt.title('Boxplot of Age and Fare (before outlier handling)')
    plt.savefig('boxplot_before.png')  # Save the plot
    plt.close()
    
    # Handle outliers in Fare using IQR method
    Q1 = df['Fare'].quantile(0.25)
    Q3 = df['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap the outliers
    df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, 
                         np.where(df['Fare'] < lower_bound, lower_bound, df['Fare']))
    
    # Visualize after handling outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['Age', 'Fare']])
    plt.title('Boxplot of Age and Fare (after outlier handling)')
    plt.savefig('boxplot_after.png')
    plt.close()
    
    return df

def main():
    # Step 1: Load and explore data
    print("Loading and exploring data...")
    df = load_and_explore_data()
    
    # Step 2: Handle missing values
    print("\nHandling missing values...")
    df = handle_missing_values(df)
    
    # Step 3: Encode categorical features
    print("\nEncoding categorical features...")
    df = encode_categorical_features(df)
    
    # Step 4: Scale numerical features
    print("\nScaling numerical features...")
    df = scale_numerical_features(df)
    
    # Step 5: Handle outliers
    print("\nHandling outliers...")
    df = handle_outliers(df)
    
    # Save cleaned data
    df.to_csv('cleaned_titanic.csv', index=False)
    print("\nCleaned data saved to 'cleaned_titanic.csv'")
    
    # Print final info
    print("\n=== Final Dataset Info ===")
    print(df.info())
    
    print("\nData cleaning and preprocessing completed successfully!")

if __name__ == "__main__":
    main()