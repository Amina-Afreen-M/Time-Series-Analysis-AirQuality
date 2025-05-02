# Air Quality Analysis and Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from datetime import datetime, timedelta
import warnings
import os

os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

warnings.filterwarnings('ignore')

plt.style.use('ggplot')
sns.set(style="whitegrid")

# ---  Load Data ---
try:
    df = pd.read_excel('AirQualityUCI.xlsx')
    print("Data shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("File not found. Using sample data for demonstration.")
    
    dates = pd.date_range('2004-03-10 18:00:00', periods=9357, freq='H')
    df = pd.DataFrame({
        'Date': [d.date() for d in dates],
        'Time': [d.time() for d in dates],
        'CO(GT)': np.random.normal(2, 1, 9357),
        'PT08.S1(CO)': np.random.normal(1100, 200, 9357),
        'NMHC(GT)': np.random.normal(270, 75, 9357),
        'C6H6(GT)': np.random.normal(10, 4, 9357),
        'PT08.S2(NMHC)': np.random.normal(900, 100, 9357),
        'NOx(GT)': np.random.normal(180, 50, 9357),
        'PT08.S3(NOx)': np.random.normal(800, 80, 9357),
        'NO2(GT)': np.random.normal(100, 30, 9357),
        'PT08.S4(NO2)': np.random.normal(1000, 150, 9357),
        'PT08.S5(O3)': np.random.normal(1000, 200, 9357),
        'T': np.random.normal(18, 8, 9357),
        'RH': np.random.normal(50, 15, 9357),
        'AH': np.random.normal(1, 0.4, 9357)
    })
    for col in df.columns[2:]:
        mask = np.random.choice([True, False], size=df.shape[0], p=[0.05, 0.95])
        df.loc[mask, col] = -200

# --- Data Preprocessing ---

print("\nMissing values (before preprocessing):")
print(df.isnull().sum())

print("\nCount of -200 values in each column:")
for col in df.columns:
    if df[col].dtype in [np.int64, np.float64]:
        print(f"{col}: {(df[col] == -200).sum()}")

df.replace(-200, np.nan, inplace=True)

print("\nPercentage of missing values in each column:")
missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage)


print("\nDate and Time columns format:")
print(df['Date'].head())
print(df['Time'].head())

if isinstance(df['Date'].iloc[0], str):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
elif not isinstance(df['Date'].iloc[0], pd.Timestamp):
    df['Date'] = pd.to_datetime(df['Date'])

if isinstance(df['Time'].iloc[0], str):
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
elif not hasattr(df['Time'].iloc[0], 'hour'):  
    df['Time'] = pd.to_datetime(df['Time']).dt.time

df['Timestamp'] = pd.to_datetime(df.apply(
    lambda row: f"{row['Date'].strftime('%Y-%m-%d')} {str(row['Time'])}", 
    axis=1
))

df.drop(['Date', 'Time'], axis=1, inplace=True)

df.set_index('Timestamp', inplace=True)

if not df.index.is_monotonic_increasing:
    df = df.sort_index()



# interpolation for continuous variables
print("\nHandling missing values with interpolation...")
df = df.interpolate(method='time')


print("\nMissing values after interpolation:")
print(df.isnull().sum())


df = df.fillna(method='bfill').fillna(method='ffill')


print("\nFinal missing values check:")
print(df.isnull().sum())


print("\nBasic statistics after preprocessing:")
print(df.describe())

# --- Exploratory Data Analysis (EDA) ---

def plot_time_series(dataframe, columns, figsize=(15, 10), save_path=None):
    """Plot time series for multiple columns"""
    fig, axes = plt.subplots(len(columns), 1, figsize=figsize, sharex=True)
    
    if len(columns) == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        axes[i].plot(dataframe.index, dataframe[col])
        axes[i].set_title(f'Time Series: {col}')
        axes[i].set_ylabel(col)
        axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig

#  time series for the pollutants

pollutant_cols = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
                 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)']

fig = plot_time_series(df, pollutant_cols, figsize=(15, 20), save_path='figures/pollutant_time_series.png')
plt.suptitle('Time Series of Air Pollutants', fontsize=16)
plt.subplots_adjust(top=0.95)
plt.close(fig)

#  environmental variables
#  temperature, relative humidity, absolute humidity
env_cols = ['T', 'RH', 'AH']
fig = plot_time_series(df, env_cols, figsize=(15, 10), save_path='figures/environmental_time_series.png')
plt.suptitle('Time Series of Environmental Variables', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.close(fig)

#correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Air Quality Variables', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

#  data distribution and potential outliers using boxplots
plt.figure(figsize=(16, 12))
df.boxplot()
plt.xticks(rotation=90)
plt.title('Boxplots of Air Quality Variables', fontsize=16)
plt.tight_layout()
plt.savefig('figures/boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

# Daily patterns
def add_time_features(df):
    """Add time-based features to the dataframe"""
    df_copy = df.copy()
    df_copy['hour'] = df_copy.index.hour
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['year'] = df_copy.index.year
    df_copy['dayofyear'] = df_copy.index.dayofyear
    return df_copy

df_time = add_time_features(df)

#  hourly patterns for CO(GT)
plt.figure(figsize=(14, 7))
sns.boxplot(x='hour', y='CO(GT)', data=df_time)
plt.title('Hourly Pattern for CO(GT)', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/co_hourly_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

#  weekly patterns for CO(GT)
plt.figure(figsize=(14, 7))
sns.boxplot(x='dayofweek', y='CO(GT)', data=df_time)
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.title('Weekly Pattern for CO(GT)', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/co_weekly_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# Monthly patterns for CO(GT)
plt.figure(figsize=(14, 7))
sns.boxplot(x='month', y='CO(GT)', data=df_time)
plt.title('Monthly Pattern for CO(GT)', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/co_monthly_pattern.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Stationarity Testing ---
print("\nStationarity Testing:")

def test_stationarity(timeseries, column_name):
    """Test stationarity using Augmented Dickey-Fuller test"""

    timeseries = timeseries.dropna()
    
    # Dickey-Fuller test
    result = adfuller(timeseries)
    
    print(f'ADF Statistic for {column_name}: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
  
    if result[1] <= 0.05:
        print(f"{column_name} is stationary (reject H0)")
    else:
        print(f"{column_name} is non-stationary (fail to reject H0)")
    print("-" * 50)
    
    return result[1] <= 0.05


stationarity_results = {}
for col in pollutant_cols + env_cols:
    stationarity_results[col] = test_stationarity(df[col], col)

plt.figure(figsize=(12, 6))
plt.bar(stationarity_results.keys(), [1 if val else 0 for val in stationarity_results.values()])
plt.xticks(rotation=90)
plt.yticks([0, 1], ['Non-Stationary', 'Stationary'])
plt.title('Stationarity Test Results (ADF Test)')
plt.tight_layout()
plt.savefig('figures/stationarity_results.png', dpi=300, bbox_inches='tight')
plt.close()

# --- 5. Feature Engineering ---
print("\nPerforming Feature Engineering...")

# Create lag features
def create_lag_features(df, columns, lag_periods=[1, 2, 3, 24, 48, 72, 168]):
    """Create lag features for specified columns"""
    df_copy = df.copy()
    
    for col in columns:
        for lag in lag_periods:
            df_copy[f'{col}_lag_{lag}'] = df_copy[col].shift(lag)
    
    return df_copy


def create_rolling_features(df, columns, windows=[6, 12, 24, 48]):
    """Create rolling mean and std features for specified columns"""
    df_copy = df.copy()
    
    for col in columns:
        for window in windows:
            df_copy[f'{col}_roll_mean_{window}'] = df_copy[col].rolling(window=window).mean()
            df_copy[f'{col}_roll_std_{window}'] = df_copy[col].rolling(window=window).std()
    
    return df_copy


def add_cyclical_features(df):
    """Add cyclical time features"""
    df_copy = df.copy()
    
   
    df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy.index.hour / 24)
    df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy.index.hour / 24)
    
    
    df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy.index.dayofweek / 7)
    df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy.index.dayofweek / 7)
    
    
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy.index.month / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy.index.month / 12)
    
    return df_copy


df_features = df.copy()
df_features = add_cyclical_features(df_features)
df_features = create_lag_features(df_features, pollutant_cols + env_cols)
df_features = create_rolling_features(df_features, pollutant_cols + env_cols)

df_features = df_features.dropna()

print(f"Original dataframe shape: {df.shape}")
print(f"Feature-engineered dataframe shape: {df_features.shape}")

# --- Splitting the data for training and testing ---

test_size = int(len(df) * 0.1)
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

print(f"\nTraining data shape: {train_df.shape}")
print(f"Testing data shape: {test_df.shape}")
print(f"Training period: {train_df.index.min()} to {train_df.index.max()}")
print(f"Testing period: {test_df.index.min()} to {test_df.index.max()}")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df.index, df['CO(GT)'], label='Full Dataset')
ax.axvline(x=test_df.index.min(), color='r', linestyle='--', label='Train-Test Split')
ax.set_title('Train-Test Split Visualization (CO(GT))')
ax.legend()
plt.savefig('figures/train_test_split.png', dpi=300, bbox_inches='tight')
plt.close()

# ---  Model Building and Evaluation ---

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, column_name):
    """Calculate RMSE and R2 score for model evaluation"""
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE for {column_name}: {rmse:.4f}")
    print(f"R2 Score for {column_name}: {r2:.4f}")
    return rmse, r2


performance_results = pd.DataFrame(columns=['SARIMA_RMSE', 'SARIMA_R2', 'Prophet_RMSE', 'Prophet_R2'])

# --- ARIMA/SARIMA Modeling ---
print("\n7.1 ARIMA/SARIMA Modeling:")

def fit_sarima(train_series, test_series, column_name, order=(1,1,1), seasonal_order=(1,1,1,24)):
    """Fit SARIMA model, forecast, evaluate, and plot results"""
    print(f"\nFitting SARIMA model for {column_name}...")
    
    try:
        model = SARIMAX(train_series, 
                        order=order, 
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False)
        
        
        forecast = model_fit.forecast(steps=len(test_series))
        
       
        rmse, r2 = evaluate_model(test_series, forecast, f"{column_name} (SARIMA)")
        
        # forecast vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(train_series.index[-500:], train_series[-500:], label='Training Data')
        plt.plot(test_series.index, test_series, label='Actual Test Data')
        plt.plot(test_series.index, forecast, label='SARIMA Forecast')
        plt.title(f'SARIMA Forecast for {column_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/sarima_forecast_{column_name.replace("(", "_").replace(")", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # residual analysis
        residuals = pd.Series(model_fit.resid, index=train_series.index)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot residuals
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].grid(True)
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[0, 1], lags=40)
        axes[0, 1].set_title('ACF of Residuals')
        
        # histogram of residuals
        axes[1, 0].hist(residuals, bins=30)
        axes[1, 0].set_title('Histogram of Residuals')
        
        # QQ plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('QQ Plot of Residuals')
        
        plt.tight_layout()
        plt.suptitle(f'Residual Analysis for {column_name} - SARIMA Model', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.savefig(f'figures/sarima_residuals_{column_name.replace("(", "_").replace(")", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return model_fit, forecast, rmse, r2
    
    except Exception as e:
        print(f"Error fitting SARIMA model for {column_name}: {e}")
        return None, None, None, None



# --- Prophet Modeling ---
print("\n7.2 Prophet Modeling:")

def fit_prophet(train_df, test_df, column_name):
    """Fit Prophet model, forecast, evaluate, and plot results"""
    print(f"\nFitting Prophet model for {column_name}...")
    
   
    prophet_df = pd.DataFrame({
        'ds': train_df.index,
        'y': train_df[column_name]
    })
    
   
    prophet_df = prophet_df.dropna()
    
  
    try:
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
       
        for regressor in ['T', 'RH', 'AH']:
            if regressor in train_df.columns and not train_df[regressor].isna().any():
                model.add_regressor(regressor)
                prophet_df[regressor] = train_df[regressor]
        
        model.fit(prophet_df)
        
        
        future = pd.DataFrame({'ds': test_df.index})
        
        
        for regressor in ['T', 'RH', 'AH']:
            if regressor in test_df.columns and regressor in prophet_df.columns:
                future[regressor] = test_df[regressor].values
        
        
        forecast = model.predict(future)
        
       
        eval_df = pd.DataFrame({'y_true': test_df[column_name], 'y_pred': forecast['yhat'].values}, index=test_df.index)
        eval_df.dropna(inplace=True)
        rmse, r2 = evaluate_model(eval_df['y_true'], eval_df['y_pred'], f"{column_name} (Prophet)")
        
        # forecast vs actual
        plt.figure(figsize=(12, 6))
        plt.plot(train_df.index[-500:], train_df[column_name][-500:], label='Training Data')
        plt.plot(test_df.index, test_df[column_name], label='Actual Test Data')
        plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast')
        plt.fill_between(forecast['ds'], 
                         forecast['yhat_lower'], 
                         forecast['yhat_upper'], 
                         color='grey', 
                         alpha=0.2)
        plt.title(f'Prophet Forecast for {column_name}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/prophet_forecast_{column_name.replace("(", "_").replace(")", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig = model.plot_components(forecast)
        plt.tight_layout()
        plt.savefig(f'figures/prophet_components_{column_name.replace("(", "_").replace(")", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return model, forecast, rmse, r2
    
    except Exception as e:
        print(f"Error fitting Prophet model for {column_name}: {e}")
        return None, None, None, None




print("\n8. Feature Importance Analysis (using Correlation):")

# correlation with target variables
def analyze_feature_importance(df, target_cols):
    """Analyze feature importance using correlation"""
    for target in target_cols:
        
        correlations = df.corrwith(df[target]).sort_values(ascending=False)
        
        
        correlations = correlations.drop(target)
        
        print(f"\nTop correlated features for {target}:")
        print(correlations.head(10))
        
        
        plt.figure(figsize=(10, 6))
        correlations.head(10).plot(kind='bar')
        plt.title(f'Top Correlated Features for {target}')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'figures/correlations_{target.replace("(", "_").replace(")", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return correlations


feature_importance = {}
for col in pollutant_cols:
    feature_importance[col] = analyze_feature_importance(df, [col])


print("\n9. Applying models to all variables:")


sarima_order = (1, 1, 1)  # Default (p, d, q) order
sarima_seasonal_order = (1, 1, 1, 24)  # Default seasonal order with 24-hour seasonality


sarima_models = {}
prophet_models = {}
sarima_forecasts = {}
prophet_forecasts = {}


for col in pollutant_cols + env_cols:
    print(f"\n--- Processing {col} ---")
    
    # Fit SARIMA model
    model_fit, forecast_vals, rmse, r2 = fit_sarima(
        train_df[col],
        test_df[col],
        col,
        order=sarima_order,
        seasonal_order=sarima_seasonal_order
    )
    
    if rmse is not None:
        sarima_models[col] = model_fit
        sarima_forecasts[col] = forecast_vals
        performance_results.loc[col, 'SARIMA_RMSE'] = rmse
        performance_results.loc[col, 'SARIMA_R2'] = r2
        
    # Fit Prophet model
    model, forecast_df, rmse, r2 = fit_prophet(
        train_df,
        test_df,
        col
    )
    
    if rmse is not None:
        prophet_models[col] = model
        prophet_forecasts[col] = forecast_df
        performance_results.loc[col, 'Prophet_RMSE'] = rmse
        performance_results.loc[col, 'Prophet_R2'] = r2


print("\nFinal Performance Results (RMSE and R2 Score):")
print(performance_results)


performance_results.to_csv('results/performance_results.csv')
print("\nPerformance results saved to results/performance_results.csv")


print("\n10. Model Performance Comparison (based on RMSE):")

# comparison plot using RMSE
rmse_results = performance_results[['SARIMA_RMSE', 'Prophet_RMSE']].copy()
rmse_results.columns = ['SARIMA', 'Prophet'] # Rename for plotting clarity

plt.figure(figsize=(12, 6))
rmse_results.plot(kind='bar')
plt.title('RMSE Comparison: SARIMA vs Prophet')
plt.xlabel('Variable')
plt.ylabel('RMSE')
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


print("\nRMSE Comparison Table:")
print(rmse_results)


print("\n11. Exporting Results...")


rmse_results.to_csv('results/rmse_results.csv')


forecast_summary = pd.DataFrame()
for col in pollutant_cols + env_cols:
    if col in sarima_forecasts and col in prophet_forecasts:
        
        actual = test_df[col]
        
        
        sarima_pred = sarima_forecasts[col]
        prophet_pred = prophet_forecasts[col]['yhat']
        
        # Calculate metrics
        summary = pd.DataFrame({
            'Variable': [col, col],
            'Model': ['SARIMA', 'Prophet'],
            'RMSE': [rmse_results.loc[col, 'SARIMA'], rmse_results.loc[col, 'Prophet']],
            'R2': [r2_score(actual, sarima_pred), r2_score(actual, prophet_pred)]
        })
        
        forecast_summary = pd.concat([forecast_summary, summary])


forecast_summary.to_csv('results/forecast_summary.csv', index=False)

print("\nForecast Summary:")
print(forecast_summary)

print("\nAnalysis completed successfully!")
print("Results and figures have been saved to the 'results' and 'figures' directories.")

# --- Generate Submission File ---
print("\n12. Generating submission file (submission.xlsx)...")
try:
    
    last_date = df.index.max()
    
    
    forecast_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=48, freq='H')
    
    
    all_forecasts = pd.DataFrame(index=forecast_dates)
    
    
    all_forecasts['Date'] = all_forecasts.index.date
    all_forecasts['Time'] = all_forecasts.index.time
    
    # Generate forecasts for each variable using SARIMA models
    print("Generating 48-hour forecasts for all variables...")
    
    
    column_order = ['Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 
                   'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 
                   'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
    
   
    for col in column_order[2:]:  
        try:
           
            if col in sarima_models and sarima_models[col] is not None:
                model_fit = sarima_models[col]
            else:
                # Fit a new SARIMA model if needed
                print(f"Fitting new SARIMA model for {col}...")
                model = SARIMAX(df[col], 
                                order=sarima_order, 
                                seasonal_order=sarima_seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                sarima_models[col] = model_fit
            
            # Generate forecast for the next 48 hours
            forecast = model_fit.forecast(steps=48)
            
            all_forecasts[col] = forecast.values
            
            print(f"Forecast for {col} generated successfully.")
        except Exception as e:
            print(f"Error generating forecast for {col}: {e}")
            
            all_forecasts[col] = df[col].iloc[-1]
    
    all_forecasts = all_forecasts[column_order]
    
    all_forecasts['Date'] = all_forecasts['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
  
    all_forecasts['Time'] = all_forecasts['Time'].apply(lambda x: x.strftime('%H:%M:%S'))
    
    # write to submission.xlsx
    all_forecasts.to_excel('submission.xlsx', index=False)
    print("submission.xlsx created successfully with 48-hour forecasts for all variables.")
    
    
    print("\nFirst few rows of the submission file:")
    print(all_forecasts.head())
    
except Exception as e:
    print(f"Error creating submission.xlsx: {e}")
    
   
    try:
       
        basic_submission = pd.DataFrame(columns=['Date', 'Time'] + pollutant_cols + env_cols)
        basic_submission.to_excel('submission.xlsx', index=False)
        print("Created a basic submission.xlsx file with column headers only.")
    except Exception as e2:
        print(f"Error creating basic submission.xlsx: {e2}")

print("\n--- Analysis Script Finished ---")