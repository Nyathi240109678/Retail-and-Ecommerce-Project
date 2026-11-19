# PipeGuard Technologies
<h1 align="center">
  PipeGuard Technologies
  <br>
  AI-Powered Water Pipeline Leak Detection and Monitoring
</h1>

<p align="center">
  Intelligent water leak detection for improved water management and reduced water losses.
</p>

<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Random%20Forest-blue" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Python-Programming-yellow?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Artificial%20Intelligence-AI-white" alt="Artificial Intelligence">
  <img src="https://img.shields.io/badge/NLP-Chatbot-purple" alt="Natural Language Processing">
  <img src="https://img.shields.io/badge/Industry-Water%20Management-blue" alt="Water Management">
</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Project Objectives](#project-objectives)
- [Machine Learning Approach](#machine-learning-approach)
- [Technologies Used](#technologies-used)
- [Model Evaluation](#model-evaluation)
- [Key Features](#key-features)
- [Water Leak Assistant Chatbot](#water-leak-assistant-chatbot)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Project](#how-to-run-the-project)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Limitations](#limitations)
- [References](#references)

---

## Project Overview

PipeGuard Technologies is an AI-powered water pipeline leak detection project developed in Python. It applies machine learning to identify possible leaks in water distribution networks by analysing historical sensor readings.

The project focuses on supporting South African municipalities in reducing water losses, improving pipeline monitoring and maintaining a more reliable water supply.

PipeGuard uses a Random Forest classification model to analyse pressure, flow, tank-level and water-demand readings. It classifies network conditions as either **Leak Detected (1)** or **No Leak Detected (0)**.

The project also includes a Water Leak Assistant Chatbot, which helps users understand detection results, check the latest detection status and prepare maintenance reports.

## Problem Statement

Water leaks in underground pipelines are a major challenge for municipalities. They cause significant water wastage, increased maintenance costs and interruptions to water supply.

Leaks can remain undetected for extended periods because manual inspections cannot easily identify unusual changes in pipeline pressure, flow rates and tank levels across an entire water distribution network.

PipeGuard addresses this problem by using machine learning to analyse sensor readings and identify patterns that may indicate significant water leaks.

## Proposed Solution

PipeGuard uses a supervised machine learning approach to detect possible water pipeline leaks.

A Random Forest classification model is trained using historical water network data from the BattLeDIM 2018 dataset. The model analyses sensor readings and calculates the probability of an active leak.

The system classifies network conditions and activates an alarm after three consecutive leak predictions to help reduce false alarms.

An additional Water Leak Assistant Chatbot uses Natural Language Processing to help users interpret detection results and take appropriate follow-up action.

## Project Objectives

- Obtain and prepare historical water network sensor and leakage data from the BattLeDIM 2018 dataset.
- Analyse pressure, flow, tank-level and water-demand readings to identify patterns associated with significant leaks.
- Develop a Random Forest classification model to classify network conditions.
- Evaluate model performance using accuracy, precision, recall, F1-score and ROC-AUC.
- Apply feature engineering to capture changes in sensor readings over time.
- Require three consecutive leak predictions before activating an alarm.
- Develop a Water Leak Assistant Chatbot to help users understand detection results and prepare maintenance reports.
- Demonstrate how AI can support municipalities in reducing water losses and improving maintenance planning.

## Machine Learning Approach

**Learning type:** Supervised Learning

**Problem type:** Binary Classification

**Algorithm:** Random Forest Classifier

**Number of trees:** 200

**Dataset:** BattLeDIM 2018

**Target variable:** Leak status

The model classifies water network conditions into two categories:

- **1 – Leak Detected:** At least one pipe has a leakage value greater than 10.0.
- **0 – No Leak Detected:** No pipe has a leakage value greater than 10.0.

### Input Features

The model uses the following types of information:

- Pressure sensor readings.
- Flow sensor readings.
- Tank-level readings.
- Total network water-demand readings.
- One-hour rolling averages.
- One-hour rolling standard deviations.
- Deviations from the 24-hour rolling average.
- Hour of the day.
- Day of the week.

### Model Training

The dataset is divided chronologically:

- **60% Training Data:** Earlier historical readings used to train the model.
- **40% Testing Data:** Later historical readings used to evaluate the model.

A probability threshold of 0.50 is applied to classify readings as leak or no leak.

The model also uses a three-consecutive-reading alarm rule to reduce false alarms.

The trained model and its feature list are saved using Joblib for later use.

Saved model files:

- `leak_model_2018_with_demand.pkl`
- `leak_model_2018_with_demand_features.pkl`

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualisation |
| Joblib | Saving and loading the trained model |
| CountVectorizer | Converting chatbot questions into numerical features |
| Multinomial Naive Bayes | Classifying chatbot questions |
| Git | Version control |
| GitHub | Project hosting and collaboration |
| Excel | Working with CSV data |
| Python IDE | Writing and executing Python code |

## Model Evaluation

The Random Forest model is evaluated using the final 40% of the historical dataset, containing 41,934 five-minute timestamps from August to the end of 2018.

The following evaluation metrics are used:

- **Accuracy:** Measures the overall percentage of correct predictions.
- **Precision:** Measures how many predicted leaks were actual leaks.
- **Recall:** Measures how many actual leaks were correctly detected.
- **F1-score:** Measures the balance between precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between leak and no-leak conditions.
- **Confusion Matrix:** Shows correct predictions, false alarms and missed leaks.

### Model Performance Results

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 93.3% |
| Precision | 94.4% |
| Recall | 87.6% |
| F1-score | 90.9% |
| Baseline Accuracy | 61.8% |

### Confusion Matrix Results

| Result | Number of Readings |
|---|---:|
| True Negatives | 25,107 |
| True Positives | 14,012 |
| False Positives | 828 |
| False Negatives | 1,987 |

The model achieves an accuracy of 93.3%, outperforming the 61.8% baseline accuracy. These results demonstrate its ability to identify significant leak patterns in the historical test data.

## Key Features

- AI-powered water pipeline leak detection.
- Random Forest classification model.
- Analysis of pressure, flow, tank-level and water-demand readings.
- Time-series feature engineering.
- Leak probability calculation.
- Leak and no-leak classification.
- Three-consecutive-reading alarm confirmation.
- Live simulation using historical sensor readings.
- CSV-based leak detection.
- Detection results saved to a CSV file.
- Water Leak Assistant Chatbot.
- Maintenance report preparation and local CSV storage.

## Water Leak Assistant Chatbot

The Water Leak Assistant Chatbot is an additional feature developed using Natural Language Processing and machine learning.

It uses **CountVectorizer** to convert user questions into numerical features and a **Multinomial Naive Bayes classifier** to identify the type of question and provide an appropriate response.

The chatbot assists users with:

- **Water leak detection:** Provides information about detected leaks and guidance on what action to take.
- **Latest detection status:** Displays the latest leak status, alarm condition, detection probability and timestamp by reading `leak_alerts_output.csv`.
- **Maintenance alerts:** Allows users to prepare and save maintenance reports when a leak is detected.
- **Pressure and flow problems:** Provides information about unusual pressure and flow readings.
- **Leak prevention:** Provides guidance on identifying and preventing water leaks.
- **Water conservation:** Offers suggestions for reducing water losses.
- **Emergency guidance:** Provides basic guidance on responding to suspected leaks.
- **System information:** Explains how the detection system and its sensors work.

Maintenance reports are saved locally in a CSV file for follow-up by the maintenance team. They are not automatically sent.

## Installation and Setup

### Prerequisites

Install the following on your computer:

- Python
- Git (if cloning the repository)
- A Python IDE, such as Visual Studio Code or PyCharm

### 1. Obtain the Repository

Download or clone the PipeGuard project repository from GitHub.

### 2. Open the Project Folder

Open the downloaded project folder in your Python IDE or terminal.

### 3. Install the Required Libraries

Run the following command:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

The chatbot uses Scikit-learn, which is included in the installation above.

### 4. Check the Project Files

Ensure that the required dataset, Python scripts and saved model files are available in the project folder.

## How to Run the Project

### Run the Main Leak Detection System

Open and execute the main leak detection Python script in your IDE.

The program provides options to:

1. Run a live simulation.
2. Create a sample sensor input file.
3. Detect leaks from a sensor file.
4. View information about how the system works.
5. Quit the program.

The live simulation uses historical 2018 readings to demonstrate how the system processes sensor data and generates leak predictions.

### Run the Water Leak Assistant Chatbot

Open a terminal in the project folder and execute:

```bash
python water_leak_assisstant_chatbot.py
```

The chatbot allows users to type questions about water leak detection, maintenance alerts, pressure and flow problems, leak prevention and other supported topics.

It can also read the latest detection results from `leak_alerts_output.csv`.

## Project Structure

The following are the main project files and outputs:

| File | Description |
|---|---|
| `water_leak_assisstant_chatbot.py` | Water Leak Assistant Chatbot |
| `leak_model_2018_with_demand.pkl` | Saved trained Random Forest model |
| `leak_model_2018_with_demand_features.pkl` | Saved model feature list |
| `leak_alerts_output.csv` | Stores leak probabilities, readings and alarm statuses |
| `maintenance_alerts.csv` | Stores prepared maintenance reports |
| `README.md` | Project introduction and setup guide |
| Main leak detection Python script | Runs the leak detection system and its available options |
| BattLeDIM 2018 dataset files | Historical water network sensor and leakage data |
| Project documentation | Business Analysis 3.2 project documentation |

## Documentation

The project documentation provides further information about the development and business analysis of PipeGuard Technologies.

It covers:

- Introduction and relevance to the project theme.
- Business background.
- Problem definition.
- Proposed solution.
- Project aim and objectives.
- Business objectives and success criteria.
- Hardware, software and data requirements.
- Machine learning techniques.
- Time-series analysis.
- Natural Language Processing and the chatbot.
- Model evaluation and accuracy improvement.
- Project constraints and risks.
- References and supporting information.

## Limitations

- The current implementation uses historical 2018 data rather than a direct connection to real-time water sensors.
- The model identifies whether a significant leak is active but does not identify the exact leaking pipe.
- Detection depends on the quality and availability of sensor readings.
- The system assumes sensor readings are collected at five-minute intervals.
- Leakage values below or equal to 10.0 are treated as normal background behaviour.
- The model's performance on historical simulated data may differ from performance on a real municipal water network.
- Maintenance reports prepared through the chatbot are saved locally and are not automatically sent to the maintenance team.

---

## References

Breiman, L. (2001) 'Random forests', *Machine Learning*, 45(1), pp. 5–32.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825–2830.

Vrachimis, S.G. et al. (2020) *Battle of the Leakage Detection and Isolation Methods (BattLeDIM)*. Zenodo. Available at: https://doi.org/10.5281/zenodo.4017659.

Carnie, T. (2026) 'Leaking pipes, filthy rivers and health risks exposed in SA's latest water report cards', *Daily Maverick*, 31 March.

Parliamentary Monitoring Group (2026) *Green, Blue and No Drop Reports 2025: Ministry briefing*. Portfolio Committee on Water and Sanitation, 9 June.
