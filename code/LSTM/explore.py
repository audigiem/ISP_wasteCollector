import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from pathlib import Path

# Load JSON data
def load_data(filepath):
    """Load JSON data from file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

# Convert JSON to DataFrame
def json_to_dataframe(data):
    """Convert GeoJSON feature collection to pandas DataFrame"""
    features = data['features']
    
    records = []
    for feature in features:
        record = {
            'longitude': feature['geometry']['coordinates'][0],
            'latitude': feature['geometry']['coordinates'][1],
            'latestFullness': feature['properties']['latestFullness'],
            'reason': feature['properties']['reason'],
            'serialNumber': feature['properties']['serialNumber'],
            'description': feature['properties']['description'],
            'position': feature['properties']['position'],
            'ageThreshold': feature['properties']['ageThreshold'],
            'fullnessThreshold': feature['properties']['fullnessThreshold'],
            'timestamp': feature['properties']['timestamp']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Display basic statistics
def display_statistics(df):
    """Display basic statistics about the data"""
    print("\n=== BASIC STATISTICS ===")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique bins: {df['serialNumber'].nunique()}")
    print(f"\nFullness statistics:")
    print(df['latestFullness'].describe())
    print(f"\nReason distribution:")
    print(df['reason'].value_counts())
    print(f"\nMissing values:")
    print(df.isnull().sum())

# Plot fullness distribution
def plot_fullness_distribution(df):
    """Plot distribution of bin fullness levels"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Histogram
    axes[0, 0].hist(df['latestFullness'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Bin Fullness Levels')
    axes[0, 0].set_xlabel('Fullness Level')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot by reason
    df.boxplot(column='latestFullness', by='reason', ax=axes[0, 1])
    axes[0, 1].set_title('Fullness Distribution by Reason')
    axes[0, 1].set_xlabel('Reason')
    axes[0, 1].set_ylabel('Fullness Level')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45)
    
    # Time series of average fullness
    daily_avg = df.groupby(df['timestamp'].dt.date)['latestFullness'].mean()
    axes[1, 0].plot(daily_avg.index, daily_avg.values, linewidth=1)
    axes[1, 0].set_title('Average Daily Fullness Over Time')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Average Fullness')
    axes[1, 0].grid(True, alpha=0.3)
    plt.sca(axes[1, 0])
    plt.xticks(rotation=45)
    
    # Fullness vs Threshold
    axes[1, 1].scatter(df['fullnessThreshold'], df['latestFullness'], alpha=0.3)
    axes[1, 1].set_title('Fullness vs Fullness Threshold')
    axes[1, 1].set_xlabel('Fullness Threshold')
    axes[1, 1].set_ylabel('Latest Fullness')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fullness_distribution.png', dpi=300, bbox_inches='tight')
    print("\nSaved: fullness_distribution.png")
    plt.show()

# Plot correlations
def plot_correlations(df):
    """Plot correlation matrix for numeric variables"""
    # Select numeric columns
    numeric_cols = ['latestFullness', 'ageThreshold', 'fullnessThreshold', 
                    'longitude', 'latitude']
    
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numeric Variables')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved: correlation_matrix.png")
    plt.show()

# Seasonal decomposition
def plot_seasonal_decomposition(df, bin_id=None):
    """Plot seasonal decomposition (trend, seasonality, residuals)"""
    # Aggregate daily data
    if bin_id:
        df_bin = df[df['serialNumber'] == bin_id].copy()
        title_suffix = f" (Bin {bin_id})"
    else:
        df_bin = df.copy()
        title_suffix = " (All Bins Average)"
    
    # Create daily time series
    daily_series = df_bin.groupby(df_bin['timestamp'].dt.date)['latestFullness'].mean()
    daily_series.index = pd.to_datetime(daily_series.index)
    
    # Fill missing dates with forward fill
    daily_series = daily_series.asfreq('D', method='ffill')
    
    # Perform seasonal decomposition
    try:
        # Use additive model, period=7 for weekly seasonality
        decomposition = seasonal_decompose(daily_series, model='additive', period=7)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Original
        axes[0].plot(decomposition.observed)
        axes[0].set_title(f'Original Time Series{title_suffix}')
        axes[0].set_ylabel('Fullness')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(decomposition.trend)
        axes[1].set_title(f'Trend Component{title_suffix}')
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(decomposition.seasonal)
        axes[2].set_title(f'Seasonal Component{title_suffix}')
        axes[2].set_ylabel('Seasonality')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(decomposition.resid)
        axes[3].set_title(f'Residual Component{title_suffix}')
        axes[3].set_ylabel('Residuals')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'seasonal_decomposition{"_bin_" + str(bin_id) if bin_id else ""}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.show()
        
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")
        print("Time series might be too short or have too many missing values")

# Geographic visualization
def plot_geographic_distribution(df):
    """Plot geographic distribution of bins"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # All bins location
    axes[0].scatter(df['longitude'], df['latitude'], 
                   c=df['latestFullness'], cmap='YlOrRd', 
                   alpha=0.6, s=20)
    axes[0].set_title('Geographic Distribution of Bins (colored by fullness)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='Fullness Level')
    
    # Average fullness by location
    avg_by_location = df.groupby(['longitude', 'latitude'])['latestFullness'].mean().reset_index()
    scatter = axes[1].scatter(avg_by_location['longitude'], 
                             avg_by_location['latitude'],
                             c=avg_by_location['latestFullness'], 
                             cmap='YlOrRd', s=100, edgecolors='black')
    axes[1].set_title('Average Fullness by Bin Location')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1], label='Average Fullness')
    
    plt.tight_layout()
    plt.savefig('geographic_distribution.png', dpi=300, bbox_inches='tight')
    print("Saved: geographic_distribution.png")
    plt.show()

# Main execution
def main():
    """Main function to run all analyses"""
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'wyndham_smartbin_filllevel.json'
    print(f"Loading data from: {data_path}")
    
    data = load_data(data_path)
    print(f"Loaded {len(data['features'])} features")
    
    # Convert to DataFrame
    df = json_to_dataframe(data)
    
    # Display statistics
    display_statistics(df)
    
    # Plot fullness distribution
    plot_fullness_distribution(df)
    
    # Plot correlations
    plot_correlations(df)
    
    # Plot geographic distribution
    plot_geographic_distribution(df)
    
    # Seasonal decomposition for all bins
    plot_seasonal_decomposition(df)
    
    # Seasonal decomposition for a specific bin (select one with most data)
    bin_counts = df['serialNumber'].value_counts()
    if len(bin_counts) > 0:
        most_common_bin = bin_counts.index[0]
        print(f"\nAnalyzing bin {most_common_bin} (most frequent)")
        plot_seasonal_decomposition(df, bin_id=most_common_bin)
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()