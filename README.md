# Data Cleaning & Preprocessing Task

This repository contains my solution for Task 1 of the AI & ML Internship program.

## Task Description
The task involved cleaning and preprocessing the Titanic dataset to prepare it for machine learning.

## Steps Performed
1. **Data Exploration**: Examined the dataset structure, missing values, and basic statistics.
2. **Missing Value Handling**: 
   - Filled missing Age values with median
   - Filled missing Embarked values with mode
   - Dropped the Cabin column due to excessive missing values
3. **Categorical Encoding**:
   - Used label encoding for Sex (binary)
   - Used one-hot encoding for Embarked (multi-category)
4. **Feature Scaling**:
   - Standardized Age and Fare features
5. **Outlier Handling**:
   - Visualized outliers using boxplots
   - Capped Fare outliers using IQR method

## Files
- `data_cleaning.py`: Main Python script
- `titanic.csv`: Original dataset
- `cleaned_titanic.csv`: Processed dataset
- `boxplot_before.png`: Boxplot before outlier handling
- `boxplot_after.png`: Boxplot after outlier handling

## How to Run
1. Install requirements: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Download the Titanic dataset and save as `titanic.csv` in the project folder
3. Run the script: `python data_cleaning.py`