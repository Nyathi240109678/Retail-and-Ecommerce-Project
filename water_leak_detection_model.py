# WATER LEAK DETECTION WITH ML, we are using BattLeDIM 2018 dataset
# Model: Random Forest Classifier
# 60% TRAINING / 40% TESTING (chronological split, NOT random), all from 2018
#
# LABEL: "leak active" = at least one single pipe leaking above LEAK_THRESHOLD.
# Smaller leaks below the threshold are treated as normal background behaviour.



# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# SETTINGS

# A single pipe leaking MORE than this value counts as a (significant) leak.
LEAK_THRESHOLD = 10.0

# True  = also use the demand data (water used by customers)
# False = use only pressure, flow and tank level
USE_DEMANDS = True

# Data is every 5 minutes: 12 steps = 1 hour, 288 steps = 24 hours
STEPS_1H = 12
STEPS_24H = 288

YEAR = 2018


# 2. FUNCTIONS TO LOAD AND PREPARE THE DATA

folder = os.path.dirname(os.path.abspath(__file__))


def parse_timestamps(text):
    """
    The files use TWO different date formats:
      2018 files and 2019 SCADA files : 2019-01-31 00:05:00  (year-month-day)
      2019_Leakages.csv               : 31.01.2019 00:05     (day.month.year)
    Guessing the format would swap days and months in the 2019 leak file,
    so each format is parsed explicitly.
    """
    parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(text, format="%d.%m.%Y %H:%M", errors="coerce")
    return parsed


def load_scada(file_name, prefix):
    """Load one BattLeDIM CSV (';' separator, ',' decimal)."""
    path = os.path.join(folder, file_name)
    data = pd.read_csv(path, sep=";", decimal=",")
    data["Timestamp"] = parse_timestamps(data["Timestamp"])

    # Safety check: no timestamp may be silently lost or misread
    if data["Timestamp"].isna().any():
        raise SystemExit(f"STOP: some timestamps in {file_name} could not be read.")

    data = data.set_index("Timestamp")
    # Prefix so sensor names (n1, n4...) don't clash between files
    return data.add_prefix(prefix + "_")


def make_features(sensors):
    """
    Raw sensor readings + rolling statistics that capture pressure drops.
    Only looks at the PAST, so no future information leaks in.
    """
    parts = [sensors]

    rolling_1h = sensors.rolling(STEPS_1H)
    parts.append(rolling_1h.mean().add_suffix("_mean_1h"))
    parts.append(rolling_1h.std().add_suffix("_std_1h"))

    # How far the reading is from its normal 24-hour level
    dev = sensors - sensors.rolling(STEPS_24H).mean()
    parts.append(dev.add_suffix("_dev_24h"))

    features = pd.concat(parts, axis=1)

    # Time-of-day features (demand follows a daily cycle)
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek

    return features


# 3. LOAD DATASET

print("=" * 60)
print("LOADING DATASET (", YEAR, ")")
print("=" * 60)

pressures = load_scada(f"{YEAR}_SCADA_Pressures.csv", "pressure")
flows = load_scada(f"{YEAR}_SCADA_Flows.csv", "flow")
levels = load_scada(f"{YEAR}_SCADA_Levels.csv", "level")
leakages = load_scada(f"{YEAR}_Leakages.csv", "leak")

# Demand = how much water customers are using. High demand also lowers pressure,
# so it can help the model tell "people are using more water" from "a pipe is leaking".
# We add the TOTAL demand of the whole network (the sum of all demand nodes).
# Caution: real utilities often do not measure demand at every node in real time.

sensors = pd.concat([pressures, flows, levels], axis=1, join="inner").sort_index()

# Safety check: the leak labels must line up exactly with the sensor rows
if not leakages.index.equals(sensors.index):
    raise SystemExit("STOP: the leak timestamps do not match the sensor timestamps.")

if USE_DEMANDS:
    demands = load_scada(f"{YEAR}_SCADA_Demands.csv", "demand")
    if not demands.index.equals(sensors.index):
        raise SystemExit("STOP: the demand timestamps do not match the sensor timestamps.")
    sensors["total_demand"] = demands.sum(axis=1)
    print("Demand data added: total network demand from", demands.shape[1], "nodes")

print("Pressure sensors :", pressures.shape[1])
print("Flow sensors     :", flows.shape[1])
print("Tank level sensors:", levels.shape[1])
print("Pipes with leak data:", leakages.shape[1])

print("\nTime range:", sensors.index.min(), "->", sensors.index.max())
print("Number of rows:", len(sensors))
print("Missing values in sensor data:", sensors.isnull().sum().sum())


# 4. LEAK SIZE PER PIPE

print(f"\nLeaks in {YEAR} (max size and % of year active):")
for col in leakages.columns:
    active = leakages[col] > 0
    print(" ", col,
          "| max:", round(leakages[col].max(), 2),
          "| active %:", round(active.mean() * 100, 1),
          "| above", LEAK_THRESHOLD, "%:",
          round((leakages[col] > LEAK_THRESHOLD).mean() * 100, 1))


# 5. CREATE THE TARGET (LEAK / NO LEAK)

biggest_leak = leakages.max(axis=1)
y = (biggest_leak > LEAK_THRESHOLD).astype(int)
y = y.reindex(sensors.index).fillna(0).astype(int)

print("\nTarget: leak_active (1 = a pipe leaking above", LEAK_THRESHOLD, ", 0 = no leak)")
print(y.value_counts().rename({0: "No leak", 1: "Leak"}))
print("Leak percentage:", round(y.mean() * 100, 2), "%")

print("\nLeak % per month:")
print((y.groupby(y.index.month).mean() * 100).round(1).to_string())

if y.nunique() < 2:
    raise SystemExit("The label has only one class. Change LEAK_THRESHOLD.")


# 6. CREATE FEATURES

X = make_features(sensors)

print("\nFeatures created:", X.shape[1])
print("  - raw sensor readings:", sensors.shape[1])
print("  - rolling mean / std (1 hour), deviation from 24h average")
print("  - hour, day_of_week")


# 7. REMOVE ROWS WITH MISSING VALUES
# The first 24 hours have no 24h rolling average, so they are dropped.

valid_rows = X.notnull().all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

print("Rows kept:", len(X))


# 8. SPLIT DATA (CHRONOLOGICAL)
# First 60% of the year = training, last 40% of the year = testing.

split_index = int(len(X) * 0.60)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print("Total records   :", len(X))
print("Training records:", len(X_train), "(", X_train.index.min(), "->", X_train.index.max(), ")")
print("Testing records :", len(X_test), "(", X_test.index.min(), "->", X_test.index.max(), ")")
print("Leak % in training:", round(y_train.mean() * 100, 2))
print("Leak % in testing :", round(y_test.mean() * 100, 2))

# SAFETY CHECK: both sets must contain both classes,
# otherwise the scores would be meaningless.
if y_train.nunique() < 2:
    raise SystemExit("STOP: the training period has only one class. Change LEAK_THRESHOLD.")

if y_test.nunique() < 2:
    raise SystemExit("STOP: the testing period has only one class. Change LEAK_THRESHOLD.")


# 9. CREATE RANDOM FOREST MODEL

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


# 10. TRAIN MODEL

print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training complete.")


# 11. MAKE PREDICTIONS

print("\nMaking predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions complete.")


# 12. EVALUATE MODEL
# RECALL = leaks we catch, PRECISION = how many alarms are real.

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC AUC  :", round(auc, 4))

baseline = max(y_test.mean(), 1 - y_test.mean())
print("\nBaseline accuracy (always predict the majority class):", round(baseline, 4))
print("Your model must beat this to be useful.")


# 13. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

print("\nConfusion matrix:")
print("True negatives  (no leak, predicted no leak):", tn)
print("False positives (false alarms)              :", fp)
print("False negatives (missed leaks)              :", fn)
print("True positives  (leak found)                :", tp)


# 14. SHOW ACTUAL VS PREDICTED VALUES

results = pd.DataFrame({
    "Actual Leak": y_test.values,
    "Predicted Leak": y_pred,
    "Leak Probability": np.round(y_prob, 3),
}, index=y_test.index)

print("\nFirst 10 predictions:")
print(results.head(10))


# 15. GRAPH 1: CONFUSION MATRIX

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (2018, last 40%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["No leak", "Leak"])
plt.yticks([0, 1], ["No leak", "Leak"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)

plt.colorbar()
plt.tight_layout()
plt.show()


# 16. GRAPH 2: LEAK PROBABILITY OVER TIME

plt.figure(figsize=(12, 5))

plt.plot(results.index, results["Leak Probability"],
         color="blue", linewidth=0.8, label="Predicted leak probability")

plt.fill_between(results.index, 0, results["Actual Leak"],
                 color="red", alpha=0.25, label="Actual leak active")

plt.title("Leak Probability Over Time (2018 test period)")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 17. FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)

print(importance.head(15).to_string(index=False))


# 18. GRAPH 3: FEATURE IMPORTANCE

top = importance.head(15).iloc[::-1]

plt.figure(figsize=(9, 6))
plt.barh(top["Feature"], top["Importance"], color="teal")
plt.title("Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# 19. FINAL SUMMARY

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print("Model: Random Forest Classifier")
print("Data: 2018 only")
print("Training data: first 60% of 2018")
print("Testing data : last 40% of 2018")

print("\nTarget: leak_active (any single pipe leaking above", LEAK_THRESHOLD, ")")

print("\nFinal results:")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC AUC  :", round(auc, 4))



# 20. SAVE TRAINED MODEL

model_path = os.path.join(folder, "leak_model_2018_with_demand.pkl")
features_path = os.path.join(folder, "leak_model_2018_with_demand_features.pkl")

joblib.dump(model, model_path)
joblib.dump(X.columns.tolist(), features_path)

print("\n" + "=" * 60)
print("MODEL FILES SAVED SUCCESSFULLY")
print("=" * 60)

print("Model saved to:", model_path)
print("Features saved to:", features_path)


# 21. FINISH

print("\nProgram finished successfully.")
