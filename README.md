# Hybrid EV Residual Value Prediction System

A hybrid machine learning system combining **Kalman filtering**, **Bayesian inference**, and **gradient boosting (CatBoost)** to predict residual values of electric vehicles (EVs) with high accuracy. The system specifically focuses on Hyundai EV models and achieves significant improvements over traditional linear depreciation models.

## Overview

This project implements a three-stage prediction pipeline for electric vehicle residual values:

1.  **Kalman Filter (State-Space Model)**: Provides baseline predictions using time-series state-space modeling with realistic EV depreciation curves.
2.  **Bayesian Inference**: Estimates uncertainty in depreciation rates using Metropolis-Hastings MCMC sampling.
3.  **CatBoost Regression**: Corrects residuals by incorporating technical EV specifications and market factors.

The hybrid approach reduces prediction error (MAPE) from **8.3%** (linear model) to **2.84%** (hybrid system), representing a ~66% improvement.

## Features

-   **Realistic EV Depreciation Modeling**: Incorporates actual EV market depreciation patterns.
-   **Technical Specifications**: Considers battery size, WLTP range, energy consumption, and peak power.
-   **Market Factors**: Accounts for electricity prices and EV demand fluctuations.
-   **Interactive Selection**: User-friendly CLI for selecting vehicle models and lease terms.
-   **Statistical Validation**: Includes paired t-tests demonstrating statistical significance.
-   **Visualization**: Generates plots for predictions, confidence intervals, and feature importance.

## Project Structure

-   **`main_portal_simulation.py`**: Main entry point orchestrating the complete prediction workflow.
-   **`kalman_filter.py`**: Kalman filter implementation (State-Space Model).
-   **`bayessche_Inferenz.py`**: Bayesian inference module using Metropolis-Hastings algorithm.
-   **`catboost_model.py`**: CatBoost regression model for residual correction.
-   **`data_utils.py`**: Utility functions for data loading, selection, and test data generation.
-   **`hyundai_ev_restwerte.py`**: Synthetic dataset generator for realistic EV residual values.

## Requirements

Install dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn catboost
```


## Quick Start Guide (GitHub Codespaces)

Follow these steps to run the project directly in your browser using GitHub Codespaces.

### Step 1: Fork the Repository

1. Click the **Fork** button in the top-right corner of the repository page.
2. Create the fork under your own account.

### Step 2: Launch Codespaces

1. Click the green **Code** button on your forked repository.
2. Select the **Codespaces** tab.
3. Click **Create codespace on main**.

### Step 3: Install Extensions and Dependencies

1. Install the **Python** extension by Microsoft from the Extensions sidebar.
2. Open the terminal and run:

```bash
pip install pandas numpy scipy matplotlib seaborn catboost
```


### Step 4: Generate Data \& Run

1. **Generate the dataset:**

```bash
python src/hyundai_ev_restwerte.py
```

*This creates `hyundai_ev_restwerte.csv` containing the sample data.*

2. **Run the prediction pipeline:**

```bash
python src/main_portal_simulation.py
```


**Outputs:**

- **`plots/`**: Contains visualization images (residual value curves, feature importance charts).
- **`catboost_info/`**: Contains training logs from the CatBoost model.


## Technical Details

### Kalman Filter

Implements a linear state-space model based on Neusser (2016):

- **State vector**: [residual_value, depreciation_rate]
- **Depreciation curve**: Non-linear step function (approx. 25% year 1, 15% year 2, 10% year 3+).


### Bayesian Inference

- **Method**: Metropolis-Hastings MCMC sampling.
- **Iterations**: 50,000 (with burn-in and thinning).
- **Goal**: Quantify the uncertainty of the depreciation rate parameter $\beta$.


### CatBoost Model

- **Features**: 14 total (Brand, Model, Variant, Battery, Mileage, Region, etc.).
- **Configuration**: Depth 9, Learning rate 0.04, Loss function RMSE.
- **Purpose**: Captures non-linear dependencies that the Kalman filter misses.


## References \& Data Sources

### Core Methodologies \& Algorithms

* **Neusser, K. (2016).** *Time Series Econometrics* (2nd ed.). Springer.
* **Kalman, R. E. (1960).** A new approach to linear filtering and prediction problems. *Transactions of the ASME--Journal of Basic Engineering*.
* **Hastings, W. K. (1970).** Monte Carlo sampling methods using Markov chains. *Biometrika*.
* **Chib, S., \& Greenberg, E. (1995).** Understanding the Metropolis-Hastings Algorithm. *The American Statistician*.
* **Gelman, A., et al. (2003).** *Bayesian Data Analysis* (3rd ed.). CRC Press.
* **Prokhorenkova, L., et al. (2018).** CatBoost: unbiased boosting with categorical features. *NeurIPS*.
* **Friedman, J. H. (2002).** Stochastic gradient boosting. *Computational Statistics and Data Analysis*.


### Statistical Validation

* **Hyndman, R. J., \& Koehler, A. B. (2006).** Another look at measures of forecast accuracy. *International Journal of Forecasting*.
* **Chai, T., \& Draxler, R. R. (2014).** Root mean square error (RMSE) or mean absolute error (MAE)? *Geoscientific Model Development*.
* **Shier, R. (2004).** *Paired t-tests*. Statstutor.
* **Kim, T. K. (2015).** T-test as a parametric statistic. *Korean Journal of Anesthesiology*.


### Market Data \& EV Context

* **Clean Energy Wire (2025).** *Germany reports a 27 percent drop in electric car sales in 2024*.
* **Mobility Portal EU (2025).** *EV sales in Germany plummeted by 27%*.
* **MDPI Applied Sciences (2025).** *Renewable Energy and Price Stability*.
* **eCarsTrade (2025).** *EV Depreciation Rate - Do Electric cars hold their value?*
* **DAT / ADAC (2025).** *Elektroauto gebraucht kaufen: Darauf sollten Sie achten - Restwertanalyse*.
* **AADA (2025).** *Automotive Insights Report: Electric Vehicle Depreciation Analysis*.


### Hyundai Vehicle Specifications

* **Hyundai Motor Deutschland (2025).** *KONA Elektro | Technische Daten und Preise*.
* **EV Supply (2025).** *Hyundai IONIQ 5 (2025): Preise, Ausstattung \& Pakete*.
* **Hyundai Motor Deutschland (2025).** *IONIQ 6 Varianten \& Preise*.
* **Hyundai Motor Deutschland (2023).** *Neuer Hyundai IONIQ 5 N startet ab 74.900 Euro*.


## License

This project is provided as-is for educational and research purposes.
