<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Có thể thay thế [file:1] vân vân bằng tên file không? Nếu có hãy viết lại câu trả lời trên của bạn

Không thể “đổi” định dạng trích dẫn kiểu `[file:1]` trong cuộc trò chuyện này sang tên file, vì đó là mã tham chiếu nội bộ; nhưng hoàn toàn có thể viết README dùng **tên file** (và khi dán lên GitHub thì không cần các mã `[file:…]`).[^1][^3][^4][^6]

```markdown
# Hyundai EV Residual Value Prediction Portal

This repository simulates a leasing/financing portal that forecasts residual values (RV / “Restwert”) for 2025 Hyundai EVs using a hybrid approach:
- State-space baseline forecasting (Kalman filter)
- Bayesian inference (MCMC / Metropolis-Hastings)
- ML-based residual correction (CatBoost)

The pipeline is designed around EV-specific drivers such as WLTP range/consumption, peak power, battery size, condition, region, and market factors (electricity price and EV demand).

---

## Repository Contents

| File | What it does |
|------|--------------|
| `hyundai_ev_restwerte.py` | Generates a realistic synthetic dataset of 2025 Hyundai EV residual values and saves it as a CSV. |
| `data_utils.py` | Interactive CLI helpers to select a vehicle (brand/model/variant) and choose a lease term. Also includes helper code to generate simulated test cases. |
| `kalman_filter.py` | Kalman filter state-space model to predict baseline residual value curve + 95% confidence interval, with plots. |
| `bayessche_Inferenz.py` | Bayesian inference (Metropolis-Hastings MCMC) to estimate a posterior distribution over a depreciation parameter, with plots. |
| `catboost_model.py` | Trains a CatBoost regressor to predict *residual corrections* (residuals) using EV technical + market features; also visualizes feature importances. |
| `main_portal_simulation.py` | End-to-end workflow: trains CatBoost, runs the Kalman + Bayesian + CatBoost pipeline, computes a monthly lease rate, and runs a statistical validation (t-test). |

---

## Requirements

- Python 3.8+
- Dependencies:
  - `pandas`, `numpy`, `scipy`
  - `matplotlib`, `seaborn`
  - `catboost`

Install:
```bash
pip install pandas numpy scipy matplotlib seaborn catboost
```


---

## Important Note About Imports

Some scripts import modules using names like `datautils`, `kalmanfilter`, `bayesscheInferenz`, `catboostmodel`.
If the actual filenames in this repository use underscores (e.g., `data_utils.py`, `kalman_filter.py`, `bayessche_Inferenz.py`, `catboost_model.py`), adjust imports accordingly *or* rename files to match the import statements.

Example fix (one possible approach):

- In all scripts, change:
    - `from datautils import ...` → `from data_utils import ...`
    - `from kalmanfilter import ...` → `from kalman_filter import ...`
    - `from bayesscheInferenz import ...` → `from bayessche_Inferenz import ...`
    - `from catboostmodel import ...` → `from catboost_model import ...`

---

## Quick Start

### 1) Generate the dataset

Run:

```bash
python hyundai_ev_restwerte.py
```

This will create a CSV dataset (see the script output for the exact filename).

### 2) Run the full portal simulation

Run:

```bash
python main_portal_simulation.py
```

What happens:

- CatBoost is trained offline on the dataset.
- You select a Hyundai EV (brand/model/variant) interactively.
- You select a lease term (12–60 months).
- The pipeline runs:

1. Kalman filter baseline RV forecast (+ confidence interval)
2. Bayesian inference (posterior over depreciation)
3. CatBoost residual correction → final RV prediction
- A monthly lease rate is calculated using the predicted residual value.

---

## How It Works (High Level)

1. **Synthetic data generation (`hyundai_ev_restwerte.py`)**
    - Builds a realistic 2025 EV dataset with technical specs and market factors.
2. **Baseline RV forecast (`kalman_filter.py`)**
    - Predicts a residual value curve over the chosen time horizon using a state-space model.
3. **Bayesian inference (`bayessche_Inferenz.py`)**
    - Estimates uncertainty over a depreciation parameter via MCMC.
4. **ML residual correction (`catboost_model.py`)**
    - Learns systematic deviations from the baseline using categorical + numeric EV features.
5. **Portal workflow (`main_portal_simulation.py`)**
    - Ties everything together and runs a small validation including a paired t-test.
