# Hybrid EV Residual Value Prediction System

A hybrid machine learning system combining Kalman filtering, Bayesian inference, and gradient boosting (CatBoost) to predict residual values of electric vehicles (EVs) with high accuracy. The system specifically focuses on Hyundai EV models and achieves significant improvements over traditional linear depreciation models.

## Overview

This project implements a three-stage prediction pipeline for electric vehicle residual values:

1. **Kalman Filter (State-Space Model)**: Provides baseline predictions using time-series state-space modeling with realistic EV depreciation curves (25% year 1, 18% year 2, 12% year 3+)
2. **Bayesian Inference**: Estimates uncertainty in depreciation rates using Metropolis-Hastings MCMC sampling
3. **CatBoost Regression**: Corrects residuals by incorporating technical EV specifications and market factors

The hybrid approach reduces prediction error (MAPE) from 8.3% (linear model) to 2.84% (hybrid system), representing a ~66% improvement.

## Features

- **Realistic EV Depreciation Modeling**: Incorporates actual EV market depreciation patterns
- **Technical Specifications**: Considers battery size, WLTP range, energy consumption, peak power
- **Market Factors**: Accounts for electricity prices and EV demand fluctuations
- **Interactive Selection**: User-friendly CLI for selecting vehicle models and lease terms
- **Statistical Validation**: Includes paired t-tests demonstrating statistical significance
- **Visualization**: Generates plots for predictions, confidence intervals, and feature importance

## Project Structure

- **`main_portal_simulation.py`**: Main entry point orchestrating the complete prediction workflow
- **`kalman_filter.py`**: Kalman filter implementation with Neusser (2016) state-space notation
- **`bayessche_Inferenz.py`**: Bayesian inference module using Metropolis-Hastings algorithm
- **`catboost_model.py`**: CatBoost regression model for residual correction
- **`data_utils.py`**: Utility functions for data loading, selection, and test data generation
- **`hyundai_ev_restwerte.py`**: Synthetic dataset generator for realistic EV residual values

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
catboost
```

Install dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn catboost
```


## Usage

### 1. Generate Dataset

First, generate the synthetic EV residual value dataset:

```bash
python hyundai_ev_restwerte.py
```

This creates `hyundai_ev_restwerte.csv` with 2,000 samples of Hyundai EV data including:

- Models: KONA Elektro, IONIQ 5, IONIQ 5 N, IONIQ 6
- Features: Battery size, WLTP range/consumption, peak power, mileage, condition, market factors


### 2. Run Complete Prediction Pipeline

Execute the full workflow:

```bash
python main_portal_simulation.py
```

The system will:

1. Train the CatBoost model offline
2. Prompt you to select a vehicle brand, model, and variant
3. Ask for lease term duration (12/24/36/48/60 months)
4. Run Kalman filter prediction
5. Perform Bayesian inference
6. Apply CatBoost residual correction
7. Display final predictions with confidence intervals
8. Show statistical validation results

### 3. Run Individual Components

**Kalman Filter only:**

```python
from kalman_filter import run_kalman_filter
from data_utils import load_and_select_data_interactive, choose_lease_term

selected_data = load_and_select_data_interactive("hyundai_ev_restwerte.csv")
lease_term = choose_lease_term()
results = run_kalman_filter(selected_data, lease_term)
```

**Bayesian Inference only:**

```python
from bayessche_Inferenz import run_bayesian_inference
from data_utils import load_and_select_data_interactive

selected_data = load_and_select_data_interactive("hyundai_ev_restwerte.csv")
bayesian_results = run_bayesian_inference(selected_data)
```


## Technical Details

### Kalman Filter (kalman_filter.py)

Implements a linear state-space model:

- **State vector**: [residual_value, depreciation_rate]
- **System matrices**: Following Neusser (2016) notation with F, A, G matrices
- **Noise parameters**: Tuned for EV market volatility (Q, R matrices)
- **Depreciation curve**: Non-linear step function (25%/18%/12% yearly)


### Bayesian Inference (bayessche_Inferenz.py)

- **Prior**: Normal distribution based on expected EV depreciation rates
- **Likelihood**: Gaussian observation model with σ=750 EUR
- **Posterior sampling**: Metropolis-Hastings with 50,000 iterations
- **Burn-in**: 20% of samples discarded, thinning factor of 10


### CatBoost Model (catboost_model.py)

**Features (14 total):**

- Categorical: Brand, Model, Variant, Body type, Color, Region, Electricity price factor, EV demand factor
- Numerical: Mileage, Battery size, Peak power, WLTP consumption, WLTP range, Condition (1-5 scale)

**Hyperparameters:**

- Iterations: 1200
- Learning rate: 0.04
- Depth: 9
- Loss function: RMSE


### Dataset (hyundai_ev_restwerte.py)

**Realistic depreciation simulation:**

- Year 1: 25% depreciation
- Year 2: 18% depreciation (of remaining value)
- Year 3+: 12% depreciation (of remaining value)

**Adjustments for:**

- Mileage deviation from 12,215 km/year average
- High performance bonus (>200 kW)
- Efficiency bonus (WLTP <14 kWh/100km)
- Range premium (>500 km)
- Body type preferences (SUV 8% premium)
- Condition rating (1-5 scale: 0.65x - 1.12x multiplier)
- Market factors (electricity price, EV demand)


## Performance Metrics

Based on 400 simulated test cases:


| Model | MAPE | RMSE |
| :-- | :-- | :-- |
| Linear Depreciation | 8.30% | ~3,200 EUR |
| Hybrid System | 2.84% | ~1,100 EUR |
| **Improvement** | **65.8%** | **65.6%** |

Statistical significance: p < 0.001 (paired t-test)

## Output Examples

**Kalman Filter Prediction:**

```
Vehicle: Hyundai IONIQ 5 RWD 84kWh
Lease term: 36 months
Initial price: 52,900 EUR
Predicted residual value: 38,450 EUR ± 1,200 EUR
Depreciation: 27.3%
```

**CatBoost Correction:**

```
Baseline (Kalman): 38,450 EUR
CatBoost adjustment: +850 EUR
Final prediction: 39,300 EUR
```

**Top Feature Importances:**

1. WLTP Range (km)
2. Battery Size (kWh)
3. Mileage (km)
4. Peak Power (kW)
5. Electricity Price Factor

## Data Format

Expected CSV structure (`hyundai_ev_restwerte.csv`):


| Column | Type | Description |
| :-- | :-- | :-- |
| FahrzeugID | int | Vehicle ID |
| Marke | str | Brand (e.g., "Hyundai") |
| Modell | str | Model (e.g., "IONIQ 5") |
| Variante | str | Variant (e.g., "RWD 84kWh") |
| Neupreis_EUR | float | New price in EUR |
| Alter_Monate | int | Age in months |
| Laufleistung_km | int | Mileage in km |
| Batteriegroesse_kWh | float | Battery capacity in kWh |
| Karosserieform | str | Body type (SUV, Limousine) |
| Spitzenleistung_kW | float | Peak power in kW |
| WLTP_Energieverbrauch_kWh_100km | float | WLTP consumption |
| WLTP_Elektrische_Reichweite_km | int | WLTP range in km |
| Farbe | str | Color |
| Region | str | Region (e.g., "DE-Nord") |
| Zustand_Skala_1_5 | int | Condition (1-5) |
| Restwert_EUR | float | Residual value in EUR |
| Wertverlust_Prozent | float | Depreciation % |
| Datum_Verkauf | str | Sale date (YYYY-MM-DD) |
| Marktfaktor_Elektropreis | str | Electricity price level |
| Marktfaktor_EVNachfrage | str | EV demand level |

## References

### Core Methodologies

**State-Space Models and Kalman Filtering:**

- Neusser, K. (2016). *Time Series Econometrics* (2nd ed.). Springer Texts in Business and Economics. ISBN 978-3-031-88838-0.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Transactions of the ASME--Journal of Basic Engineering*, 82(Series D), 35-45.

**Bayesian Inference:**

- Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.
- Chib, S., \& Greenberg, E. (1995). Understanding the Metropolis-Hastings Algorithm. *The American Statistician*, 49(4), 327-335.
- Gelman, A., Carlin, J., Stern, H., Dunson, D., Vehtari, A., \& Rubin, D. (2003). *Bayesian Data Analysis* (3rd ed.). CRC Press.

**Gradient Boosting and CatBoost:**

- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., \& Gulin, A. (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31, 6638-6648.
- Friedman, J. H. (2002). Stochastic gradient boosting. *Computational Statistics and Data Analysis*, 38(4), 367-378.


### Statistical Validation

**Forecast Accuracy Metrics:**

- Hyndman, R. J., \& Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688.
- Chai, T., \& Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)?--Arguments against avoiding RMSE in the literature. *Geoscientific Model Development*, 7(3), 1247-1250.

**Hypothesis Testing:**

- Kim, T. K. (2015). T-test as a parametric statistic. *Korean Journal of Anesthesiology*, 68(6), 540-546.
- Shier, R. (2004). *Paired t-tests*. Retrieved from https://www.statstutor.ac.uk/resources/uploaded/paired-t-test.pdf
- Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.


### Market Data and Context

**EV Market Trends:**

- Clean Energy Wire. (2025). Germany reports a 27 percent drop in electric car sales in 2024. Retrieved from https://energynews.pro/en/germany-reports-a-27-percent-drop-in-electric-car-sales-in-2024/
- Mobility Portal EU. (2025). EV sales in Germany plummeted by 27%: Is it time for a new approach? Retrieved from https://mobilityportal.eu/ev-sales-in-germany-plummeted

**EV Depreciation Studies:**

- eCarsTrade. (2025). EV Depreciation Rate - Do Electric cars hold their value? Retrieved from https://ecarstrade.com/blog/ev-depreciation-rate-do-electric-cars-hold-value/
- We Buy Any Car. (2024). Electric car depreciation: An in-depth guide. Retrieved from https://www.webuyanycar.com/electric-cars/ev-depreciation


## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure code follows the existing structure and includes appropriate documentation.

## Contact

For questions or issues, please open an issue on GitHub.
