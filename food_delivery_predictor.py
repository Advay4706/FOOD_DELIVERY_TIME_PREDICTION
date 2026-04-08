import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ==============================================
# STEP 1: Load Dataset
# ==============================================
df = pd.read_csv("train.csv")
print("Dataset loaded:", df.shape)

# ==============================================
# STEP 2: Data Cleaning
# ==============================================

# Clean target variable: "(min) 24" -> 24
df["Time_taken(min)"] = df["Time_taken(min)"].str.extract(r"(\d+)").astype(float)

# Clean weather: "conditions Sunny" -> "Sunny"
df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False).str.strip()

# Strip whitespace from all string columns
str_cols = df.select_dtypes(include="str").columns
for col in str_cols:
    df[col] = df[col].str.strip()

# Clean age and ratings
df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")

# Compute distance from coordinates (Haversine formula)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

df["Distance_km"] = haversine(
    df["Restaurant_latitude"], df["Restaurant_longitude"],
    df["Delivery_location_latitude"], df["Delivery_location_longitude"]
)

# ==============================================
# STEP 3: Select Features
# ==============================================
features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Distance_km",
    "Vehicle_condition",
    "multiple_deliveries",
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "Type_of_order",
    "Festival",
    "City"
]

df = df[features + ["Time_taken(min)"]].copy()

# Replace NaN strings with actual NaN
df.replace("NaN", np.nan, inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)
print("Rows after cleaning:", len(df))

# ==============================================
# STEP 4: Encode Categorical Columns
# ==============================================
categorical_cols = ["Weatherconditions", "Road_traffic_density",
                    "Type_of_vehicle", "Type_of_order", "Festival", "City"]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print("Encoding done.")
print(df.describe())

# ==============================================
# STEP 5: Split Data
# ==============================================
X = df[features]
y = df["Time_taken(min)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining samples:", len(X_train))
print("Testing  samples:", len(X_test))

# ==============================================
# STEP 6: Train Models
# ==============================================

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ==============================================
# STEP 7: Evaluate
# ==============================================
lr_r2   = r2_score(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

rf_r2   = r2_score(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

print("\n--- Linear Regression ---")
print(f"R2 Score : {lr_r2:.4f}")
print(f"RMSE     : {lr_rmse:.4f} minutes")

print("\n--- Random Forest ---")
print(f"R2 Score : {rf_r2:.4f}")
print(f"RMSE     : {rf_rmse:.4f} minutes")

print("\n--- Sample Predictions (Random Forest) ---")
print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
for actual, pred in zip(list(y_test[:8]), rf_preds[:8]):
    print(f"{actual:<10.1f} {pred:<10.1f} {abs(actual - pred):<10.2f}")

# ==============================================
# STEP 8: Visualizations
# ==============================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle("Food Delivery Time Prediction - ML Results", fontsize=16, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# Plot 1: Actual vs Predicted (Linear Regression)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, lr_preds, alpha=0.3, color="#4C72B0", s=10)
ax1.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit")
ax1.set_xlabel("Actual Time (min)")
ax1.set_ylabel("Predicted Time (min)")
ax1.set_title(f"Linear Regression\nR2={lr_r2:.3f}, RMSE={lr_rmse:.2f}")
ax1.legend()

# Plot 2: Actual vs Predicted (Random Forest)
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, rf_preds, alpha=0.3, color="#55A868", s=10)
ax2.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2, label="Perfect Fit")
ax2.set_xlabel("Actual Time (min)")
ax2.set_ylabel("Predicted Time (min)")
ax2.set_title(f"Random Forest\nR2={rf_r2:.3f}, RMSE={rf_rmse:.2f}")
ax2.legend()

# Plot 3: Feature Importance (Random Forest)
ax3 = fig.add_subplot(gs[1, 0])
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
importances.plot(kind="barh", ax=ax3, color="#4C72B0")
ax3.set_title("Feature Importance (Random Forest)")
ax3.set_xlabel("Importance Score")

# Plot 4: Model Comparison
ax4 = fig.add_subplot(gs[1, 1])
models = ["Linear Regression", "Random Forest"]
r2_scores = [lr_r2, rf_r2]
rmse_scores = [lr_rmse, rf_rmse]
x = np.arange(len(models))
width = 0.35
bars1 = ax4.bar(x - width/2, r2_scores, width, label="R2 Score", color="#4C72B0")
bars2 = ax4.bar(x + width/2, [r/50 for r in rmse_scores], width, label="RMSE/50", color="#DD8452")
ax4.set_xticks(x)
ax4.set_xticklabels(models)
ax4.set_title("Model Comparison")
ax4.legend()
for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", fontsize=9)

fig.text(0.5, 0.01,
         f"Linear Regression: R2={lr_r2:.4f}, RMSE={lr_rmse:.2f} min   |   Random Forest: R2={rf_r2:.4f}, RMSE={rf_rmse:.2f} min",
         ha="center", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="gray"))

plt.savefig("results.png", dpi=150, bbox_inches="tight")
print("\nVisualization saved as results.png")
