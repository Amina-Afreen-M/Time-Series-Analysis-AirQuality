# Air Quality Analysis and Forecasting Dashboard

## Project Overview

This project implements a comprehensive air quality analysis and forecasting system using time series methodologies. It provides an interactive dashboard for visualizing air quality data, analyzing temporal patterns, and generating forecasts using different models.

## Features

- **Complete Dashboard**: A comprehensive view that combines all analyses in one screen
- **Time Series Visualization**: Interactive plots of air quality variables over time
- **Temporal Pattern Analysis**: Hourly, weekly, and monthly patterns for each variable
- **Correlation Analysis**: Heatmap visualization of relationships between variables
- **Stationarity Testing**: Results of Augmented Dickey-Fuller tests for each variable
- **ACF/PACF Analysis**: Autocorrelation and partial autocorrelation function plots
- **Forecasting**: SARIMA and Prophet model forecasts with performance metrics
- **Residual Analysis**: Diagnostic plots for model validation

## Dataset

The project uses the UCI Air Quality dataset, which contains hourly averaged responses from an array of 5 metal oxide chemical sensors embedded in an Air Quality Chemical Multisensor Device. The dataset includes:

- **Pollutant Measurements**:
  - CO (Carbon Monoxide)
  - NMHC (Non-Methane Hydrocarbons)
  - C6H6 (Benzene)
  - NOx (Nitrogen Oxides)
  - NO2 (Nitrogen Dioxide)
  - O3 (Ozone) - indirectly measured via PT08.S5

- **Environmental Variables**:
  - Temperature (T)
  - Relative Humidity (RH)
  - Absolute Humidity (AH)

If the data file is not found, the dashboard will generate sample data for demonstration purposes.

## Methodology
## Data Description

The dataset consists of 9358 instances, each representing hourly averaged responses from various chemical sensors. Ground truth hourly averaged concentrations for various pollutants including CO, Non-Metanic Hydrocarbons (NMHC), Benzene, Total Nitrogen Oxides (NOx), and Nitrogen Dioxide (NO2) were provided by a co-located reference certified analyzer. Missing values are tagged with -200. 


### Data Preprocessing

- Handling missing values with interpolation and forward/backward filling
- Creating a proper timestamp index from date and time columns
- Sorting data chronologically

### Exploratory Data Analysis

- Time series visualization for all variables
- Correlation analysis between variables
- Distribution analysis and outlier detection
- Temporal pattern analysis (hourly, daily, monthly)

### Time Series Analysis

- **Stationarity Testing**: Augmented Dickey-Fuller test to determine if time series are stationary
- **ACF/PACF Analysis**: To identify potential AR and MA terms for ARIMA models

### Forecasting Models

1. **SARIMA (Seasonal AutoRegressive Integrated Moving Average)**
   - Captures both trend and seasonality in the data
   - Parameters selected based on ACF/PACF analysis

2. **Prophet**
   - Facebook's time series forecasting model
   - Handles seasonality at multiple levels (daily, weekly, yearly)
   - Robust to missing data and outliers

### Performance Metrics

- **RMSE (Root Mean Square Error)**: Measures the average magnitude of errors
- **R² Score**: Indicates the proportion of variance in the dependent variable explained by the model

## Installation

```bash
# Clone the repository
git clone https://github.com/Amina-Afreen-M/Time-Series-Analysis-AirQuality.git
cd TimeSeriesAnalysis

#q Install required packages
pip install -r requirements.txt
```

### Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- plotly
- dash
- dash-bootstrap-components
- statsmodels
- prophet
- scikit-learn

## Usage

### Running the Dashboard

```bash
python dashboard.py
```

This will start the Dash server, and you can access the dashboard by opening a web browser and navigating to `http://127.0.0.1:8050/`.

### Running the Analysis Script

```bash
python project.py
```

The `project.py` script is the core analysis engine that implements the complete data processing and modeling pipeline. Key features include:

**Data Processing:**
- Automated data loading from 'AirQualityUCI.xlsx' with fallback to synthetic data generation
- Robust handling of missing values through interpolation and forward/backward filling
- Timestamp creation and index optimization for time series analysis

**Analysis Components:**
- Comprehensive EDA with automated visualization generation
- Time series decomposition and pattern analysis
- Advanced statistical testing including stationarity analysis
- Correlation studies between pollutants and environmental variables

**Modeling and Forecasting:**
- SARIMA model implementation with automatic parameter selection
- Prophet model integration for robust forecasting
- Parallel model training for all air quality variables
- Cross-validated performance metrics (RMSE, R²)

**Output Generation:**
- Automated creation of visualization plots in the `figures` directory
- Performance metrics and forecasting results saved to `results` directory
- Comprehensive model comparison and evaluation reports

## Results

The analysis provides insights into:

1. **Temporal Patterns**: How air pollutants vary by hour of day, day of week, and month
2. **Correlations**: Relationships between different pollutants and environmental factors
3. **Forecasting Performance**: Comparison of SARIMA and Prophet models for each variable
4. **Stationarity**: Which variables exhibit stationary behavior

## Project Structure

```
TimeSeriesAnalysis/
├── AirQualityUCI.xlsx       # Main dataset file
├── dashboard.py             # Interactive Dash dashboard
├── project.py               # Main analysis script
├── read_excel.py            # Utility script for reading Excel files
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore configuration
├── LICENSE                  # MIT License file
├── README.md                # Project documentation
├── submission.xlsx          # Submission data file
├── figures/                 # Generated visualizations
│   └── *.png                # Various visualization outputs
└── results/                 # Analysis results
    └── *.csv                # Performance metrics and forecasting results
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

