import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data():
    # Load and validate the data
    try:
        df = pd.read_csv('Datasets/outbreaks.csv')
        print("\nDataset Overview:")
        print(f"Number of records: {len(df)}")
        print("\nMissing values:")
        print(df.isnull().sum())
        return df
    except FileNotFoundError:
        print("Error: The file 'Datasets/outbreak.csv' was not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def preprocess_data(df):
    # Clean and preprocess the data
    if df is None:
        return None
    
    processed_df = df.copy()
    
    # Convert Year and Month to datetime
    processed_df['Date'] = pd.to_datetime(
        processed_df['Year'].astype(str) + '-' + 
        processed_df['Month'].astype(str).str.zfill(2) + '-01'
    )
    
    # Handle missing values
    numeric_columns = ['Illnesses', 'Hospitalizations', 'Fatalities']
    processed_df[numeric_columns] = processed_df[numeric_columns].fillna(0)
    
    categorical_columns = ['State', 'Location', 'Food', 'Ingredient', 'Species', 'Serotype/Genotype', 'Status']
    for col in categorical_columns:
        processed_df[col] = processed_df[col].fillna('Unknown')
    
    return processed_df

def analyze_trends(df):
    # Analyze outbreak trends over time
    # Group by year and calculate metrics
    yearly_trends = df.groupby('Year').agg({
        'Illnesses': ['count', 'sum'],
        'Hospitalizations': 'sum',
        'Fatalities': 'sum'
    })
    
    # Flatten column names
    yearly_trends.columns = ['_'.join(col).strip() for col in yearly_trends.columns.values]
    yearly_trends = yearly_trends.reset_index()
    
    # Prepare data for regression
    X = yearly_trends['Year'].values.reshape(-1, 1)
    y = yearly_trends['Illnesses_count'].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate confidence intervals
    y_pred = model.predict(X)
    residuals = y - y_pred
    std_error = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
    ci = 1.96 * std_error
    
    print("\nQ1: Outbreak Trend Analysis")
    print(f"Annual change in outbreaks: {model.coef_[0]:.2f} ± {ci:.2f}")
    print(f"R-squared: {model.score(X, y):.3f}")
    
    trend = "increasing" if model.coef_[0] > 0 else "decreasing"
    print(f"Overall trend: Outbreaks are {trend} by {abs(model.coef_[0]):.2f} cases per year")
    
    return yearly_trends, model.coef_[0], ci

def analyze_contaminants(df):
    # Analyze impact by contaminant
    df['Pathogen'] = df['Species'] + ' - ' + df['Serotype/Genotype']
    
    pathogen_impact = df.groupby('Pathogen').agg({
        'Illnesses': ['count', 'sum'],
        'Hospitalizations': 'sum',
        'Fatalities': 'sum'
    })
    
    # Flatten column names
    pathogen_impact.columns = ['_'.join(col).strip() for col in pathogen_impact.columns.values]
    pathogen_impact = pathogen_impact.reset_index()
    
    # Calculate impact score
    pathogen_impact['Impact_Score'] = (
        pathogen_impact['Illnesses_sum'] * 1 +
        pathogen_impact['Hospitalizations_sum'] * 10 +
        pathogen_impact['Fatalities_sum'] * 100
    )
    
    # Sort and get top contaminants
    top_pathogens = pathogen_impact.sort_values('Impact_Score', ascending=False)
    
    print("\nQ2: Most Impactful Contaminants")
    print("\nTop 5 contaminants by overall impact:")
    for _, row in top_pathogens.head().iterrows():
        print(f"\nPathogen: {row['Pathogen']}")
        print(f"Total Illnesses: {row['Illnesses_sum']:,}")
        print(f"Total Hospitalizations: {row['Hospitalizations_sum']:,}")
        print(f"Total Fatalities: {row['Fatalities_sum']:,}")
        print(f"Impact Score: {row['Impact_Score']:,.2f}")
    
    return top_pathogens

def analyze_locations(df):
    # Analyze risk by location
    location_stats = df.groupby('Location').agg({
        'Illnesses': ['count', 'sum', 'mean'],
        'Hospitalizations': ['sum', 'mean'],
        'Fatalities': ['sum', 'mean']
    })
    
    # Flatten column names
    location_stats.columns = ['_'.join(col).strip() for col in location_stats.columns.values]
    location_stats = location_stats.reset_index()
    
    # Calculate risk score
    location_stats['Risk_Score'] = (
        location_stats['Illnesses_mean'] +
        location_stats['Hospitalizations_mean'] * 10 +
        location_stats['Fatalities_mean'] * 100
    )
    
    # Sort locations by risk score
    risky_locations = location_stats.sort_values('Risk_Score', ascending=False)
    
    print("\nQ3: Location Risk Analysis")
    print("\nLocations ranked by risk (top 5):")
    for _, row in risky_locations.head().iterrows():
        print(f"\nLocation: {row['Location']}")
        print(f"Average Illnesses per Outbreak: {row['Illnesses_mean']:.2f}")
        print(f"Average Hospitalizations per Outbreak: {row['Hospitalizations_mean']:.2f}")
        print(f"Average Fatalities per Outbreak: {row['Fatalities_mean']:.2f}")
        print(f"Risk Score: {row['Risk_Score']:.2f}")
    
    return risky_locations

def main():
    # Main analysis pipeline
    # Load data
    print("Loading data...")
    raw_df = load_data()
    if raw_df is None:
        return
    
    # Preprocess data
    print("\nPreprocessing data...")
    df = preprocess_data(raw_df)
    if df is None:
        return
    
    # Analyze trends
    print("\nAnalyzing trends...")
    yearly_trends, trend_slope, trend_ci = analyze_trends(df)
    
    # Analyze contaminants
    print("\nAnalyzing contaminants...")
    pathogen_results = analyze_contaminants(df)
    
    # Analyze locations
    print("\nAnalyzing locations...")
    location_results = analyze_locations(df)
    
    return {
        'yearly_trends': yearly_trends,
        'pathogen_results': pathogen_results,
        'location_results': location_results
    }

# Execute the analysis
if __name__ == "__main__":
    results = main()