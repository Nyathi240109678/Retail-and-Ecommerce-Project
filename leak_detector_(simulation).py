# WATER LEAK DETECTOR - LIVE SENSOR DEMO
# Uses the saved Random Forest model (trained on the first 60% of 2018).
#
# WHAT THIS PROGRAM SHOWS
# In a real water network, sensors send a new reading every 5 minutes.
# The detector keeps the last 24 hours of readings, turns them into the same
# 154 features the model was trained on, and the model gives a leak probability.
# If the probability stays high enough for 3 readings in a row (15 minutes),
# an alarm is raised. This avoids false alarms from one-off spikes.
#
# THE THREE DEMOS
# 1. Live simulation : replays real sensor readings one by one, as if they were
#                      arriving from the network right now.
# 2. Create a sample input file : saves sensor readings to a CSV you can open.
# 3. Detect leaks from a file : reads a CSV of sensor readings (like the sample
#                      file) and saves the results to leak_alerts_output.csv.
#
# NOTE: the readings come from the 2018 data set, so this is a replay of the
# past, not a connection to real sensors. The model was trained on the first
# 60% of 2018, so the demos only use dates from the last 40% (unseen data).


import os
import time
from collections import deque

import numpy as np
import pandas as pd
import joblib


# SETTINGS

ALARM_LEVEL = 0.50        # a reading counts as "leak" when the probability is at least this
CONFIRM_READINGS = 3      # alarm after this many leak readings in a row (3 = 15 minutes)
LEAK_THRESHOLD = 10.0     # "real leak" in the data = any pipe above this
STEPS_1H = 12             # 5-minute readings: 12 = 1 hour
STEPS_24H = 288           # 288 = 24 hours
SUFFIXES = ("_mean_1h", "_std_1h", "_dev_24h")

folder = os.path.dirname(os.path.abspath(__file__))

SAMPLE_FILE = os.path.join(folder, "sensor_readings_input.csv")
OUTPUT_FILE = os.path.join(folder, "leak_alerts_output.csv")


def close_program():
    input("\nPress Enter to close...")
    raise SystemExit


# 1. LOAD THE SAVED MODEL

print("=" * 60)
print("WATER LEAK DETECTOR - LIVE SENSOR DEMO")
print("=" * 60)

candidates = [
    ("leak_model_2018_with_demand.pkl", "leak_model_2018_with_demand_features.pkl"),
    ("leak_model_2018.pkl", "leak_model_2018_features.pkl"),
]

model = None
for model_name, features_name in candidates:
    model_path = os.path.join(folder, model_name)
    features_path = os.path.join(folder, features_name)
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        print("\nSaved model loaded:", model_name)
        break

if model is None:
    print("ERROR: no saved model was found.")
    print("Run water_leak_detection_2018_with_demand.py (or water_leak_detection_2018.py)")
    print("first, then run this program from the same folder.")
    close_program()

# The raw sensor columns the model needs (everything that is not a rolling feature)
base_sensors = [
    c for c in feature_names
    if not c.endswith(SUFFIXES) and c not in ("hour", "day_of_week")
]
needs_demand = "total_demand" in base_sensors

print("The model reads", len(base_sensors), "sensor values per reading and uses",
      len(feature_names), "features.")


# 2. THE DETECTOR (this is the part a real system would run)

def build_features(readings):
    """Turn raw sensor readings into the features the model was trained on."""
    s = readings[base_sensors].astype(float)

    parts = [s]
    rolling_1h = s.rolling(STEPS_1H)
    parts.append(rolling_1h.mean().add_suffix("_mean_1h"))
    parts.append(rolling_1h.std().add_suffix("_std_1h"))
    parts.append((s - s.rolling(STEPS_24H).mean()).add_suffix("_dev_24h"))

    features = pd.concat(parts, axis=1)
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    return features[feature_names]


class LeakDetector:
    """Receives one sensor reading at a time and says whether there is a leak."""

    def __init__(self, model, alarm_level=ALARM_LEVEL, confirm=CONFIRM_READINGS):
        self.model = model
        self.alarm_level = alarm_level
        self.confirm = confirm
        self.times = deque(maxlen=STEPS_24H)
        self.values = deque(maxlen=STEPS_24H)
        self.run_leak = 0          # leak readings in a row
        self.run_normal = 0        # normal readings in a row
        self.alarm = False         # is the alarm on?

    def warm_up(self, history):
        """Fill the 24-hour memory (a real system fills it during the first day)."""
        for t, row in zip(history.index, history[base_sensors].values):
            self.times.append(t)
            self.values.append(row)

    def update(self, timestamp, reading):
        """Add the newest reading. Returns (probability, leak_reading, alarm_on)."""
        self.times.append(pd.Timestamp(timestamp))
        self.values.append([reading[c] for c in base_sensors])

        if len(self.values) < STEPS_24H:
            return None, None, False   # still collecting the first 24 hours

        window = pd.DataFrame(list(self.values), index=list(self.times), columns=base_sensors)
        features = build_features(window).iloc[[-1]]
        self.model.set_params(n_jobs=1)
        probability = float(self.model.predict_proba(features)[0, 1])
        leak_reading = probability >= self.alarm_level

        # the alarm switches on/off only after several readings in a row
        if leak_reading:
            self.run_leak += 1
            self.run_normal = 0
        else:
            self.run_normal += 1
            self.run_leak = 0
        if not self.alarm and self.run_leak >= self.confirm:
            self.alarm = True
        if self.alarm and self.run_normal >= self.confirm:
            self.alarm = False

        return probability, leak_reading, self.alarm


def detect_dataframe(readings):
    """Run the detector on many readings at once (used for CSV files)."""
    features = build_features(readings)
    valid = features.notnull().all(axis=1)
    model.set_params(n_jobs=-1)
    probability = pd.Series(np.nan, index=readings.index)
    probability[valid] = model.predict_proba(features[valid])[:, 1]
    return probability


def apply_confirmation(flags, confirm=CONFIRM_READINGS):
    """Same alarm rule as the live detector, applied to a whole series of readings."""
    alarm, run_leak, run_normal = False, 0, 0
    out = []
    for flag in flags.values:
        if flag:
            run_leak, run_normal = run_leak + 1, 0
        else:
            run_normal, run_leak = run_normal + 1, 0
        if not alarm and run_leak >= confirm:
            alarm = True
        if alarm and run_normal >= confirm:
            alarm = False
        out.append(alarm)
    return pd.Series(out, index=flags.index)


# 3. DATA FOR THE DEMOS (2018 sensor readings)

def parse_timestamps(text):
    parsed = pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(text, format="%d.%m.%Y %H:%M", errors="coerce")
    return parsed


def load_scada(file_name, prefix):
    path = os.path.join(folder, file_name)
    if not os.path.exists(path):
        print("ERROR:", file_name, "was not found.")
        print("Put this program in the same folder as the 2018 CSV files.")
        close_program()
    data = pd.read_csv(path, sep=";", decimal=",")
    data["Timestamp"] = parse_timestamps(data["Timestamp"])
    if data["Timestamp"].isna().any():
        print("ERROR: some timestamps in", file_name, "could not be read.")
        close_program()
    return data.set_index("Timestamp").add_prefix(prefix + "_")


print("\nLoading the 2018 sensor readings for the demos (this takes a moment)...")

sensor_data = pd.concat([
    load_scada("2018_SCADA_Pressures.csv", "pressure"),
    load_scada("2018_SCADA_Flows.csv", "flow"),
    load_scada("2018_SCADA_Levels.csv", "level"),
], axis=1, join="inner").sort_index()

if needs_demand:
    demand_data = load_scada("2018_SCADA_Demands.csv", "demand")
    sensor_data["total_demand"] = demand_data.sum(axis=1).reindex(sensor_data.index)

missing = [c for c in base_sensors if c not in sensor_data.columns]
if missing:
    print("ERROR: the data is missing sensors the model needs:", missing[:5])
    close_program()

sensor_data = sensor_data[base_sensors]

leak_file = load_scada("2018_Leakages.csv", "leak")
real_leak = (leak_file.max(axis=1) > LEAK_THRESHOLD).reindex(sensor_data.index)

# The model was trained on the first 60% of the usable readings,
# so only the last 40% is "new" data for it.
usable_times = sensor_data.index[STEPS_24H - 1:]
test_start = usable_times[int(len(usable_times) * 0.60)]
last_time = sensor_data.index.max()

print("Ready. New (unseen) data runs from", test_start, "to", last_time)


# 4. HELPERS

def ask_start(default_date):
    text = input(f"Start date (YYYY-MM-DD) [{default_date}]: ").strip() or default_date
    date = pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")
    if pd.isna(date):
        print("ERROR: Invalid date. Use the format YYYY-MM-DD, for example 2018-10-06.")
        return None
    if date < test_start.normalize() + pd.Timedelta(days=1):
        print("ERROR: choose a date from", (test_start.normalize() + pd.Timedelta(days=1)).date(),
              "onwards.")
        print("Earlier dates were part of the training data, so the model has seen them.")
        return None
    if date > last_time:
        print("ERROR: the data ends on", last_time.date())
        return None
    return date


def ask_number(question, default, low, high):
    text = input(f"{question} [{default}]: ").strip() or str(default)
    try:
        value = float(text)
    except ValueError:
        print("ERROR: please enter a number.")
        return None
    if value < low or value > high:
        print(f"ERROR: please enter a number from {low} to {high}.")
        return None
    return value


# 5. DEMO 1: LIVE SIMULATION

def live_simulation():
    print("\nGood dates to try:")
    print("  2018-09-05  no leak all day (the alarm should stay off)")
    print("  2018-10-10  a leak is running all day (the alarm should stay on)")
    print("  2018-10-06  a new leak starts at 02:35 (the alarm comes about 45 minutes later)")
    date = ask_start("2018-10-10")
    if date is None:
        return
    hours = ask_number("How many hours to watch", 12, 1, 72)
    if hours is None:
        return
    delay = ask_number("Seconds between readings (0 = instant)", 0.05, 0, 2)
    if delay is None:
        return

    start_pos = sensor_data.index.searchsorted(date)
    end_pos = min(start_pos + int(hours * 12), len(sensor_data))
    history = sensor_data.iloc[start_pos - STEPS_24H:start_pos]

    detector = LeakDetector(model)
    detector.warm_up(history)

    print("\n" + "=" * 60)
    print("LIVE MONITORING - sensors reporting every 5 minutes")
    print("=" * 60)
    print("(The detector already holds the previous 24 hours of readings.)")
    print("Time              Avg pressure  Pump flow  Leak prob.  Reading   Alarm  Real leak")
    print("-" * 82)

    pressure_cols = [c for c in base_sensors if c.startswith("pressure_")]
    shown = []
    previous_state = False
    alarm_total = 0
    correct_total = 0
    first_alarm = None
    first_real = None

    for pos in range(start_pos, end_pos):
        timestamp = sensor_data.index[pos]
        reading = sensor_data.iloc[pos]

        probability, leak_reading, detected = detector.update(timestamp, reading)
        is_real = bool(real_leak.iloc[pos])

        if detected and not previous_state:
            print("*" * 82)
            print(f"*** ALARM RAISED at {timestamp:%d %b %H:%M}: possible leak in the network ***")
            print("*" * 82)
            if first_alarm is None:
                first_alarm = timestamp
        if (not detected) and previous_state:
            print("-" * 30, f"back to normal at {timestamp:%H:%M}", "-" * 30)
        previous_state = detected

        alarm_total += int(detected)
        correct_total += int(detected == is_real)
        if is_real and first_real is None:
            first_real = timestamp

        print(f"{timestamp:%Y-%m-%d %H:%M}  "
              f"{reading[pressure_cols].mean():10.2f}  "
              f"{reading['flow_PUMP_1']:9.2f}  "
              f"{probability * 100:8.1f}%  "
              f"{'LEAK' if leak_reading else 'normal':<8}  "
              f"{'ON' if detected else 'off':<5}  "
              f"{'yes' if is_real else 'no'}")

        if delay > 0:
            time.sleep(delay)

    n = end_pos - start_pos
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Readings processed        :", n)
    print("Alarm was on for          :", alarm_total, "readings", f"({alarm_total / n * 100:.1f}%)")
    print("Alarm state correct       :", correct_total, f"({correct_total / n * 100:.1f}%)")
    print("First alarm               :", first_alarm if first_alarm is not None else "none")
    print("First real leak reading   :", first_real if first_real is not None else "none in this period")
    print("(The 'Real leak' column comes from the leak file, only to check the detector.)")


# 6. DEMO 2: CREATE A SAMPLE INPUT FILE

def create_sample_file():
    print("\nThis saves sensor readings to:", os.path.basename(SAMPLE_FILE))
    print("It includes the 24 hours of history the detector needs before the chosen date.")
    date = ask_start("2018-10-06")
    if date is None:
        return
    hours = ask_number("How many hours of readings to check", 24, 1, 72)
    if hours is None:
        return

    start_pos = sensor_data.index.searchsorted(date)
    end_pos = min(start_pos + int(hours * 12), len(sensor_data))
    sample = sensor_data.iloc[start_pos - STEPS_24H:end_pos].copy()
    sample.index.name = "Timestamp"
    sample.to_csv(SAMPLE_FILE)

    print("\nSaved", len(sample), "readings (", sample.shape[1], "sensor columns ) to:")
    print(SAMPLE_FILE)
    print("Open it in Notepad (not Excel) to see what the detector receives.")
    print("Then choose option 3 to detect leaks from this file.")


# 7. DEMO 3: DETECT LEAKS FROM A FILE

def to_numbers(frame):
    for col in frame.columns:
        if frame[col].dtype == object:
            frame[col] = pd.to_numeric(frame[col].astype(str).str.replace(",", ".", regex=False),
                                       errors="coerce")
    return frame


def detect_from_file():
    text = input(f"File to check [{os.path.basename(SAMPLE_FILE)}]: ").strip().strip('"')
    path = text if text else SAMPLE_FILE
    if not os.path.isabs(path):
        path = os.path.join(folder, path)

    if not os.path.exists(path):
        print("ERROR: file not found:", path)
        return

    try:
        data = pd.read_csv(path, sep=None, engine="python")
    except Exception as error:
        print("ERROR: the file could not be read:", error)
        return

    time_col = next((c for c in data.columns if str(c).strip().lower() == "timestamp"), data.columns[0])
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    if data[time_col].isna().any():
        print("ERROR: some timestamps could not be read. Use the format 2018-10-06 02:35:00.")
        return

    data = data.set_index(time_col).sort_index()
    data.index.name = "Timestamp"

    absent = [c for c in base_sensors if c not in data.columns]
    if absent:
        print("ERROR: the file is missing", len(absent), "sensor column(s), for example:", absent[:5])
        return

    data = to_numbers(data[base_sensors].copy())
    if data.isnull().any().any():
        print("ERROR: some sensor values are missing or not numbers.")
        return

    if len(data) < STEPS_24H:
        print(f"ERROR: the file needs at least {STEPS_24H} readings (24 hours) of history;")
        print("it has", len(data), "readings.")
        return

    probability = detect_dataframe(data).dropna()
    leak_reading = probability >= ALARM_LEVEL
    detected = apply_confirmation(leak_reading)

    results = pd.DataFrame({
        "leak_probability": probability.round(4),
        "reading": np.where(leak_reading, "LEAK", "normal"),
        "alarm": np.where(detected, "ON", "off"),
    })
    results.index.name = "Timestamp"
    results.to_csv(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("DETECTION FROM FILE")
    print("=" * 60)
    print("File                      :", os.path.basename(path))
    print("Readings in the file      :", len(data))
    print("Readings checked          :", len(results),
          "(the first 24 hours are used as history)")
    print("Checked from              :", results.index.min())
    print("Checked to                :", results.index.max())
    print("Readings above alarm level:", int(leak_reading.sum()), f"({leak_reading.mean() * 100:.1f}%)")
    print("Alarm was on for          :", int(detected.sum()), "readings", f"({detected.mean() * 100:.1f}%)")

    # list the alarm periods
    run_id = (detected != detected.shift()).cumsum()
    periods = []
    for _, group in probability.groupby(run_id):
        if detected[group.index[0]]:
            periods.append((group.index[0], group.index[-1], group.max()))

    if periods:
        print("\nALARM PERIODS")
        for start, end, top in periods[:15]:
            minutes = int((end - start).total_seconds() / 60) + 5
            print(f"  {start:%Y-%m-%d %H:%M} to {end:%H:%M}  ({minutes} min)  highest probability {top * 100:.0f}%")
        if len(periods) > 15:
            print("  ...", len(periods) - 15, "more")
        print("\nDETECTOR SAYS: LEAK ALARM RAISED")
    else:
        print("\nDETECTOR SAYS: NO LEAK ALARM in this file")

    print("\nFull results saved to:", OUTPUT_FILE)


# 8. HOW IT WORKS

def explain():
    print("\n" + "=" * 60)
    print("HOW THE DETECTOR WORKS")
    print("=" * 60)
    print("1. Sensors send a reading every 5 minutes (pressure, pump flow,")
    print("   tank level" + (", total demand)." if needs_demand else ")."))
    print("2. The detector keeps the last 24 hours of readings in memory.")
    print("3. From that memory it works out", len(feature_names), "features: the reading itself,")
    print("   the average and spread over the last hour, how far each sensor")
    print("   is from its own 24-hour average, plus the hour and weekday.")
    print("4. The Random Forest (200 decision trees, trained on 2018 readings")
    print("   where the leaks were known) looks at those features. Each tree")
    print("   votes leak or no leak, and the votes give a probability.")
    print("5. A reading counts as 'leak' if the probability is at least", f"{ALARM_LEVEL:.2f}.")
    print("6. The alarm switches ON after", CONFIRM_READINGS, "leak readings in a row",
          f"({CONFIRM_READINGS * 5} minutes) and OFF after", CONFIRM_READINGS, "normal ones,")
    print("   so one-off spikes do not cause false alarms.")
    print("\nIt detects THAT a leak exists. It does not say which pipe is leaking.")
    print("The alarm level can be changed (ALARM_LEVEL at the top of this file):")
    print("lower = catches more leaks but more false alarms.")
    print("CONFIRM_READINGS controls how many readings in a row are needed.")


# 9. MENU

while True:
    print("\n")
    print("What would you like to do?")
    print("1. Live simulation (sensors reporting every 5 minutes)")
    print("2. Create a sample sensor input file")
    print("3. Detect leaks from a sensor file")
    print("4. How does it work?")
    print("5. Quit")

    choice = input("Choose 1, 2, 3, 4, or 5: ").strip()

    if choice == "1":
        live_simulation()
    elif choice == "2":
        create_sample_file()
    elif choice == "3":
        detect_from_file()
    elif choice == "4":
        explain()
    elif choice == "5":
        break
    else:
        print("ERROR: Invalid choice.")

close_program()
