# Food Delivery Time Prediction

Predicts food delivery time (in minutes) by comparing Linear Regression and Random Forest on a dataset of 45,000+ delivery records. Random Forest achieved **R² of 0.83** and **RMSE of 3.9 minutes**.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)

---

## Overview

Delivery time estimation is a core problem for food delivery platforms. Inaccurate estimates frustrate customers and hurt platform reliability. This project builds a prediction pipeline from raw GPS and operational data to a trained regression model, with full visualisation of results.

Key decisions made in this project:

- Derived a **distance feature** from raw restaurant and customer GPS coordinates using the **Haversine formula**, since distance is not directly available in the dataset
- Compared two models — **Linear Regression** (baseline) and **Random Forest** (ensemble) — to measure the performance gain from a non-linear approach
- Used **Label Encoding** for 6 categorical features including weather, traffic density, vehicle type, and city

---

## Dataset

**Source:** `train.csv` (included in repository)

**Size:** 45,000+ records

| Feature | Description |
|---|---|
| `Restaurant_latitude / longitude` | GPS coordinates of the restaurant |
| `Delivery_location_latitude / longitude` | GPS coordinates of the drop-off point |
| `Delivery_person_Age` | Age of the delivery person |
| `Delivery_person_Ratings` | Average rating of the delivery person |
| `Weatherconditions` | Weather at time of delivery (e.g. Sunny, Stormy) |
| `Road_traffic_density` | Traffic level (Low / Medium / High / Jam) |
| `Vehicle_condition` | Condition score of the delivery vehicle |
| `Type_of_vehicle` | Vehicle used (motorcycle, scooter, etc.) |
| `Type_of_order` | Order category (Snack, Meal, Drinks, Buffet) |
| `Festival` | Whether a festival was ongoing |
| `City` | City tier (Metropolitan, Urban, Semi-Urban) |
| `multiple_deliveries` | Number of deliveries in the same trip |
| `Time_taken(min)` | **Target variable** — actual delivery time in minutes |

---

## Project Workflow

### 1. Data Loading
Raw CSV loaded with Pandas. Shape and basic info printed for inspection.

### 2. Data Cleaning
- Target column `Time_taken(min)` contained strings like `"(min) 24"` — extracted the numeric value using regex
- `Weatherconditions` column had a prefix `"conditions "` stripped out
- Whitespace trimmed from all string columns
- Age, ratings, and multiple deliveries cast to numeric types with `errors='coerce'`
- Rows with missing values dropped after all cleaning steps

### 3. Feature Engineering
Distance between restaurant and customer computed using the **Haversine formula**, which accounts for the curvature of the Earth:

```
Distance (km) = 2R * arcsin(sqrt(sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)))
```

This replaces the raw latitude/longitude columns with a single, meaningful numeric feature.

### 4. Encoding
Six categorical columns encoded using `sklearn.preprocessing.LabelEncoder`:
- `Weatherconditions`
- `Road_traffic_density`
- `Type_of_vehicle`
- `Type_of_order`
- `Festival`
- `City`

### 5. Train/Test Split
80/20 split with `random_state=42` for reproducibility.

### 6. Model Training
- **Linear Regression** — baseline model, assumes a linear relationship between features and delivery time
- **Random Forest** (100 estimators, `random_state=42`) — ensemble of decision trees, captures non-linear patterns

### 7. Evaluation
Both models evaluated on the held-out test set using:
- **R² Score** — proportion of variance explained (higher is better, max 1.0)
- **RMSE** — root mean squared error in minutes (lower is better)

---

## Results

| Model | R² Score | RMSE (minutes) |
|---|---|---|
| Linear Regression | 0.41 | ~8.5 |
| **Random Forest** | **0.83** | **3.9** |

Random Forest significantly outperforms Linear Regression, confirming that delivery time has non-linear relationships with the input features.

**Top predictors (by feature importance):**
1. Distance (km)
2. Delivery person ratings
3. Road traffic density
4. Delivery person age

The `results.png` file in the repository contains four plots:
- Actual vs Predicted scatter plot for both models
- Feature importance bar chart (Random Forest)
- Side-by-side model comparison (R² and RMSE)

---

## Tech Stack

- **Python 3.x**
- **Pandas** — data loading, cleaning, feature selection
- **NumPy** — Haversine formula computation
- **Scikit-learn** — model training, Label Encoding, evaluation metrics
- **Matplotlib** — visualisation and result plots

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Advay4706/FOOD_DELIVERY_TIME_PREDICTION.git
cd FOOD_DELIVERY_TIME_PREDICTION
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib
```

**3. Run the script**
```bash
python food_delivery_predictor.py
```

This will:
- Load and clean `train.csv`
- Engineer the distance feature
- Train both models
- Print evaluation metrics to the terminal
- Save `results.png` with all visualisations

---

## File Structure

```
FOOD_DELIVERY_TIME_PREDICTION/
├── food_delivery_predictor.py   # Main script — full pipeline from data to results
├── train.csv                    # Dataset (45,000+ delivery records)
├── results.png                  # Output visualisations
└── README.md
```

---

## Author

**Advay Aggarwal**  
B.Tech Computer Science and Business Systems, VIT Vellore  
[GitHub](https://github.com/Advay4706) · [LinkedIn](https://linkedin.com/in/advay-aggarwal)
