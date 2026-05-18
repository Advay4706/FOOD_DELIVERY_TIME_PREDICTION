# 🍔 Food Delivery Time Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A machine learning project that predicts food delivery time (in minutes) using real-world delivery data — comparing Linear Regression and Random Forest models.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Visualizations](#visualizations)
- [Future Improvements](#future-improvements)

---

## Overview

Late deliveries cost restaurants customer trust and drivers their ratings. This project builds a predictive model to estimate how long a food delivery will take, based on factors like delivery agent age, ratings, distance, traffic, weather, and more.

Two regression models are trained and compared:
- **Linear Regression** — baseline model
- **Random Forest Regressor** — ensemble model for improved accuracy

---

## Dataset

**File:** `train.csv`

The dataset contains real-world food delivery records with information about delivery agents, order details, restaurant/customer locations, and environmental conditions.

| Column | Description |
|---|---|
| `Delivery_person_Age` | Age of the delivery agent |
| `Delivery_person_Ratings` | Rating of the delivery agent (out of 5) |
| `Restaurant_latitude/longitude` | GPS coordinates of the restaurant |
| `Delivery_location_latitude/longitude` | GPS coordinates of the customer |
| `Weatherconditions` | Weather at time of delivery (Sunny, Fog, etc.) |
| `Road_traffic_density` | Traffic level (Low / Medium / High / Jam) |
| `Vehicle_condition` | Condition score of delivery vehicle |
| `Type_of_order` | Order category (Snack, Meal, Drinks, etc.) |
| `Type_of_vehicle` | Vehicle used (Bicycle, Scooter, Motorcycle) |
| `multiple_deliveries` | Number of deliveries being handled simultaneously |
| `Festival` | Whether a festival is ongoing |
| `City` | City tier (Metropolitan, Urban, Semi-Urban) |
| `Time_taken(min)` | ✅ **Target variable** — delivery time in minutes |

---

## Features Used

```
Delivery_person_Age, Delivery_person_Ratings, Distance_km (computed),
Vehicle_condition, multiple_deliveries, Weatherconditions,
Road_traffic_density, Type_of_vehicle, Type_of_order, Festival, City
```

> **Note:** `Distance_km` is a derived feature computed using the **Haversine formula** from restaurant and delivery GPS coordinates.

---

## Methodology

```
Raw CSV
   │
   ▼
Data Cleaning
  ├─ Strip "conditions " prefix from Weather column
  ├─ Extract numeric value from "Time_taken(min)" (e.g. "(min) 24" → 24)
  ├─ Convert Age, Ratings, multiple_deliveries to numeric
  └─ Drop rows with NaN values
   │
   ▼
Feature Engineering
  └─ Compute Distance_km via Haversine formula
   │
   ▼
Label Encoding
  └─ Encode: Weatherconditions, Road_traffic_density,
             Type_of_vehicle, Type_of_order, Festival, City
   │
   ▼
Train / Test Split  (80% / 20%, random_state=42)
   │
   ├──► Linear Regression
   └──► Random Forest (100 estimators)
   │
   ▼
Evaluation: R² Score & RMSE
```

---

## Results

| Model | R² Score | RMSE (minutes) |
|---|---|---|
| Linear Regression | ~0.82 | ~4.5 min |
| **Random Forest** | **~0.98** | **~1.7 min** |

> Random Forest significantly outperforms Linear Regression, capturing non-linear relationships between features and delivery time.

**Output visualization saved to:** `results.png`

---

## Project Structure

```
FOOD_DELIVERY_TIME_PREDICTION/
│
├── food_delivery_predictor.py   # Main ML pipeline
├── train.csv                    # Dataset
├── results.png                  # Auto-generated visualization
└── README.md                    # Project documentation
```

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/Advay4706/FOOD_DELIVERY_TIME_PREDICTION.git
cd FOOD_DELIVERY_TIME_PREDICTION
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 3. Run the predictor

```bash
python food_delivery_predictor.py
```

This will:
- Load and clean the dataset
- Compute distances using the Haversine formula
- Train Linear Regression and Random Forest models
- Print evaluation metrics to the console
- Save visualizations to `results.png`

---

## Visualizations

The script generates a 2×2 grid of plots saved as `results.png`:

| Plot | Description |
|---|---|
| Actual vs Predicted (LR) | Scatter plot of Linear Regression performance |
| Actual vs Predicted (RF) | Scatter plot of Random Forest performance |
| Feature Importance | Horizontal bar chart of RF feature importances |
| Model Comparison | Side-by-side R² and RMSE comparison |

---

## Future Improvements

- [ ] Add XGBoost / LightGBM models for further comparison
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deploy as a REST API using Flask or FastAPI
- [ ] Build an interactive web dashboard for predictions
- [ ] Add cross-validation for more robust evaluation
- [ ] Integrate real-time traffic and weather APIs

---

## Author

**Advay** — [@Advay4706](https://github.com/Advay4706)

---

## License

This project is open source under the [MIT License](LICENSE).
