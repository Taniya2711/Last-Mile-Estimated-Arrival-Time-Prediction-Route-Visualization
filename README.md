# 📦 ETA Prediction & Route Optimization Dashboard

An end-to-end **last-mile delivery ETA prediction, uncertainty quantification, risk analysis, and route optimization system** built using Machine Learning, probabilistic modelling, geospatial analysis, and Operations Research.

---

## 🚀 Overview

This project simulates a real-world **last-mile delivery intelligence system** that predicts delivery times, quantifies uncertainty around those predictions, identifies potentially risky deliveries, and optimizes delivery routes subject to time-window constraints.

The complete pipeline combines:

* **Classical Machine Learning**

  * Linear Regression
  * Ridge Regression
  * Random Forest Regressor

* **Advanced Machine Learning**

  * XGBoost Regressor
  * LightGBM Quantile Regression
  * NGBoost

* **Uncertainty Quantification**

  * Prediction intervals using LightGBM quantile predictions
  * Predictive standard deviation using NGBoost
  * Calibration of prediction intervals
  * Prediction interval coverage analysis
  * Pinball loss and CRPS-based evaluation

* **Feature Engineering**

  * Temporal features
  * Delivery urgency
  * Distance and route features
  * Package-weight features
  * Traffic and weather features
  * Vehicle information
  * Geospatial clustering
  * Interaction features
  * Binned categorical features

* **Route Optimization**

  * Google OR-Tools
  * Vehicle Routing Problem with Time Windows (VRPTW)
  * ETA-derived delivery windows
  * Haversine/geodesic travel-time matrices
  * Feasible route generation

* **Explainable AI**

  * SHAP global feature importance
  * SHAP summary/bee-swarm plots
  * Local SHAP explanations
  * Partial Dependence Plots
  * Uncertainty attribution

* **Interactive Visualization**

  * Streamlit dashboard
  * Folium route maps
  * Geospatial delivery visualization
  * Risk-based route visualization

The project was built **from scratch using synthetic delivery data** because a suitable real-world dataset containing the required combination of delivery coordinates, time windows, vehicle information, traffic/weather conditions, and actual delivery durations was not readily available.

The objective is therefore not to claim real-world production accuracy, but to demonstrate an end-to-end **Data Science + Machine Learning + Probabilistic Modelling + Optimization + Software Engineering pipeline**.

---

# 🎯 Project Objective

The primary objective is to answer four practical questions:

### 1. 🔮 How long will a delivery take?

Predict the expected delivery duration using historical delivery characteristics such as:

* Distance
* Route duration
* Delivery urgency
* Number of previous stops
* Traffic
* Weather
* Vehicle type
* Package weight
* Priority
* Geospatial location

### 2. 📏 How uncertain is the prediction?

Instead of producing only:

> "The delivery will take 35 minutes."

the probabilistic models attempt to produce information such as:

> "The predicted ETA is 35 minutes, with an uncertainty range of approximately ±7 minutes."

This allows the system to distinguish between deliveries where the ETA is relatively reliable and deliveries where the prediction is more uncertain.

### 3. ⚠️ Which deliveries are risky?

Predictions are converted into risk levels based on the estimated uncertainty:

* **Low Risk**
* **Medium Risk**
* **High Risk**

This allows operational teams to identify deliveries that may require additional attention or scheduling buffers.

### 4. 🧭 What route should the vehicle follow?

The predicted ETA information is incorporated into a **Vehicle Routing Problem with Time Windows (VRPTW)**.

Google OR-Tools is used to find feasible delivery sequences while satisfying delivery-time constraints and minimizing travel time.

---

# 🏗️ End-to-End Pipeline

```text
Synthetic Delivery Data
          ↓
Data Cleaning & Validation
          ↓
Feature Engineering
          ↓
Geospatial Clustering
          ↓
Exploratory Data Analysis
          ↓
Baseline & Classical ML Models
          ↓
XGBoost
          ↓
LightGBM Quantile Regression
          ↓
NGBoost Probabilistic Regression
          ↓
Uncertainty Quantification
          ↓
Calibration & Coverage Analysis
          ↓
Risk Classification
          ↓
ETA-Aware Route Optimization
          ↓
Google OR-Tools VRPTW
          ↓
Folium Route Visualization
          ↓
Streamlit Dashboard
```

---

# 📊 Dataset

## Synthetic Data Generation

The dataset contains approximately **10,000 synthetic delivery orders** generated around a central depot location.

The simulated depot is located around:

```text
Latitude  : 28.6139
Longitude : 77.2090
```

representing a central location in Delhi.

Orders are generated over a **7-day period**.

Delivery locations are randomly generated within approximately a **10 km radius** of the depot.

---

## 📋 Initial Dataset Features

The generated dataset contains information such as:

| Feature                 | Description                        |
| ----------------------- | ---------------------------------- |
| `order_time`            | Time at which the order was placed |
| `delivery_window_start` | Beginning of the delivery window   |
| `delivery_window_end`   | End of the delivery window         |
| `delivery_lat`          | Delivery latitude                  |
| `delivery_lon`          | Delivery longitude                 |
| `vehicle_id`            | Assigned vehicle identifier        |
| `package_weight_kg`     | Package weight                     |
| `priority`              | High / Medium / Low                |
| `distance_km`           | Approximate distance from depot    |

Additional operational variables are subsequently generated.

---

# 🧪 Synthetic Data Realism

Since the project uses synthetic data, additional noise and inconsistencies are deliberately introduced to make the modelling problem less artificially clean.

The dataset includes simulated:

### GPS Noise

Latitude and longitude are perturbed using Gaussian noise to simulate GPS inaccuracies.

### Distance Noise

Distance values are modified by approximately ±5%.

### Package Weight Noise

Package weights receive approximately ±10% noise.

### Delivery Window Jitter

Delivery-window timestamps receive approximately ±15 minutes of jitter.

### Priority Flipping

A small percentage of priority labels are modified to simulate imperfect operational data.

### Vehicle Reassignment

A small percentage of vehicle assignments are randomly reassigned.

### Traffic

Three traffic conditions are simulated:

* Low
* Medium
* High

### Weather

Four weather conditions are simulated:

* Clear
* Rain
* Snow
* Fog

### Vehicle Type

Two vehicle types are simulated:

* EV
* GDV

### Previous Stops

The number of stops completed before a delivery is randomly generated between 0 and 5.

These additions are intended to simulate some of the imperfections found in operational delivery datasets.

---

# 🎯 Target Variable

The primary regression target is:

```text
delivery_duration_mins
```

The target represents the simulated delivery duration in minutes.

The initial route duration is calculated using:

```text
Route Duration =
Distance / Effective Speed × 60
```

where the effective speed depends on traffic conditions.

Additional delays are then introduced based on:

* Weather
* Number of previous stops
* Gaussian noise

The final target is constrained to a realistic range to avoid extreme synthetic outliers.

---

# 🧹 Data Cleaning & Preprocessing

The dataset goes through several validation and preprocessing steps.

These include:

* Missing-value checks
* Datetime conversion
* Positive-duration validation
* Geographic range validation
* Removal of invalid route durations
* Validation of delivery-window timestamps
* Duration sanity checks
* Outlier checks
* Categorical encoding
* Numerical normalization

The dataset is also checked for logical relationships between:

```text
delivery_duration_mins
route_duration_min
duration_minutes
```

This helps identify deliveries where the simulated delivery duration differs substantially from the expected route duration.

---

# 🛠️ Feature Engineering

Several features are engineered from the raw variables.

## ⏰ Temporal Features

### Order Hour

```text
order_hour
```

Extracts the hour of the day from the order timestamp.

### Day of Week

```text
order_dayofweek
```

Encodes the day of the week.

### Weekend Flag

```text
is_weekend
```

Indicates whether the order was placed on a weekend.

### Holiday Flag

```text
is_holiday
```

Uses Indian holiday information to identify holidays.

---

# ⏱️ Delivery-Time Features

### Delivery Window Duration

```text
delivery_window_duration_mins
```

Measures the length of the delivery window.

### Urgency

```text
urgency_mins
```

Measures the time between the order and the end of its delivery window.

### Total Duration

```text
duration_minutes
```

Measures the elapsed time from order placement to delivery-window end.

### Late-by Minutes

```text
late_by_minutes
```

Provides a measure of the difference between the available delivery time and the predicted/actual delivery duration.

---

# 📦 Package Features

### Package Weight

```text
package_weight_kg
```

### Weight Category

Packages are divided into:

* Very Light
* Light
* Medium
* Heavy

Both **label encoding** and **one-hot encoding** are retained for experimentation.

---

# 🚦 Traffic & Weather

Traffic is encoded into:

```text
Low
Medium
High
```

Weather is represented by:

```text
Clear
Rain
Snow
Fog
```

One-hot encoded versions are also generated while retaining the original categorical columns.

---

# 🚚 Vehicle Features

The dataset contains:

```text
vehicle_id
vehicle_type
```

Vehicle type includes:

* EV
* GDV

---

# 📍 Geospatial Feature Engineering

Delivery latitude and longitude are used to identify spatial patterns.

## K-Means Location Clustering

Delivery coordinates are standardized and clustered using **K-Means**.

The project uses:

```text
10 location clusters
```

The resulting feature is:

```text
location_cluster
```

This captures the idea that deliveries in different geographic regions may have systematically different delivery characteristics.

---

## Distance to Cluster Center

The Haversine distance between each delivery location and its corresponding cluster centroid is calculated:

```text
distance_to_cluster_center_km
```

This provides an additional measure of how centrally or remotely a delivery lies within its geographical cluster.

---

# 🔗 Interaction Features

Several interaction features are created to allow models to capture relationships between variables.

### Urgency × Distance

```text
urgency_distance_interaction
```

### Package Weight × Previous Stops

```text
weight_stops_interaction
```

### Priority × Distance

```text
priority_distance_interaction
```

### Location Cluster × Route Duration

```text
cluster_route_interaction
```

---

# 📊 Cluster-Level Features

Average characteristics are calculated for each location cluster, including:

* Average duration
* Average urgency
* Average package weight
* Average number of previous stops

For example:

```text
cluster_avg_duration
cluster_avg_urgency
cluster_avg_weight
cluster_avg_num_stops
```

These provide contextual information about the delivery area.

---

# 📦 Binning

Continuous variables are also converted into operationally meaningful ranges.

### Distance Bins

```text
0–2 km
2–5 km
5–10 km
10–20 km
20–50 km
50 km+
```

### Duration Bins

```text
0–15 min
15–30 min
30–60 min
60–90 min
90–120 min
120 min+
```

### Urgency Bins

```text
Overdue
0–30 min
30–60 min
60–120 min
120 min+
```

These are primarily useful for EDA and interpretation.

---

# 📈 Exploratory Data Analysis

A comprehensive EDA stage is included before modelling.

The analysis covers:

* Target distribution
* Histograms
* KDE distributions
* Boxplots
* Skewness
* Kurtosis
* Outlier detection
* Feature correlations
* Correlation heatmaps
* Delivery duration vs distance
* Delivery duration by hour
* Delivery duration by day of week
* Traffic effects
* Weather effects
* Package-weight effects
* Number-of-stops effects
* Spatial delivery distribution
* Location clusters
* Delivery urgency
* Delivery punctuality
* Vehicle-level delivery duration
* Distance-bin behaviour
* Urgency-bin behaviour
* Cluster-level behaviour
* Pairplots
* Multicollinearity analysis

---

# 🔍 Multicollinearity Analysis

Variance Inflation Factor (VIF) is calculated for selected numerical variables.

The analysis considers features such as:

```text
distance_km
urgency_mins
duration_minutes
route_duration_min
late_by_minutes
num_stops_before
```

This helps identify potentially redundant or highly correlated predictors before final model selection.

---

# 🌲 Feature Importance

A Random Forest model is also used to obtain an initial estimate of feature importance.

This provides an additional feature-selection perspective alongside:

* Correlation analysis
* VIF
* Domain reasoning
* Model-based importance

---

# 🤖 Machine Learning Models

The project evaluates several models progressively.

---

## 1. 📉 Baseline — Mean ETA

A simple mean prediction is used as the baseline.

Every test observation receives the mean delivery duration calculated from the dataset.

The baseline is evaluated using:

* MAE
* RMSE
* MAPE
* R²

This establishes a minimum benchmark that more sophisticated models should outperform.

---

# 2. 📐 Linear Regression

Linear Regression is used as a classical baseline model.

It assumes that the target can be approximated as a linear combination of the input features.

The model is evaluated using:

```text
MAE
RMSE
R²
```

Residual analysis is also performed.

---

# 3. 🛡️ Ridge Regression

Ridge Regression extends Linear Regression by introducing L2 regularization.

This is particularly useful when predictors are correlated.

The model uses:

```text
alpha = 1.0
```

and is evaluated using the same metrics as Linear Regression.

---

# 4. 🌳 Random Forest Regressor

Random Forest is introduced to capture:

* Non-linear relationships
* Feature interactions
* Complex relationships between delivery characteristics

Hyperparameters are tuned using `GridSearchCV`.

The search includes:

```text
n_estimators
max_depth
min_samples_split
```

---

# 5. ⚡ XGBoost

XGBoost is used as a more powerful gradient-boosted tree model.

The initial configuration includes parameters such as:

```text
n_estimators = 100
max_depth = 5
learning_rate = 0.1
```

Hyperparameter tuning is subsequently performed using `GridSearchCV`.

Parameters considered include:

```text
n_estimators
max_depth
learning_rate
subsample
```

The tuned configuration is then retrained and evaluated on a held-out test set.

---

# 📊 Model Evaluation Metrics

The regression models are evaluated using:

### MAE — Mean Absolute Error

Measures the average absolute difference between predicted and actual delivery duration.

Lower is better.

```text
MAE = average(|actual - predicted|)
```

It can be interpreted directly in minutes.

---

### RMSE — Root Mean Squared Error

Penalizes larger prediction errors more heavily than MAE.

Lower is better.

---

### R² — Coefficient of Determination

Measures the proportion of target variability explained by the model.

Higher is better.

---

### MAPE — Mean Absolute Percentage Error

Used for the baseline evaluation to provide a percentage-based error measure.

---

# 🧠 Explainable AI with SHAP

SHAP is used to understand why machine-learning models make their predictions.

The project applies SHAP analysis to tree-based models including:

* XGBoost
* Random Forest
* LightGBM
* NGBoost through model-agnostic KernelExplainer

---

## SHAP Analysis Includes

### Global Feature Importance

Identifies which features have the largest overall impact on ETA predictions.

### SHAP Summary / Beeswarm Plot

Shows:

* Feature importance
* Direction of influence
* Distribution of feature effects

### Local Explanation

Individual predictions can be investigated to understand which features pushed a particular ETA higher or lower.

---

# 🎲 Advanced Probabilistic Modelling

A major component of the project is moving beyond point predictions.

Instead of predicting only:

```text
ETA = 35 minutes
```

the system attempts to estimate:

```text
ETA ≈ 35 minutes
Uncertainty ≈ ±7 minutes
```

Two probabilistic approaches are implemented:

1. **LightGBM Quantile Regression**
2. **NGBoost**

---

# 🌲 LightGBM Quantile Regression

LightGBM is trained separately for three quantiles:

```text
10th percentile
50th percentile
90th percentile
```

These produce:

```text
eta_10
eta_50
eta_90
```

The 50th percentile represents the median ETA, while the 10th and 90th percentiles provide an estimated prediction interval.

---

## LightGBM Prediction Interval

Conceptually:

```text
Lower ETA = eta_10

Median ETA = eta_50

Upper ETA = eta_90
```

The interval width is:

```text
eta_90 - eta_10
```

A larger interval indicates greater uncertainty.

---

# 📏 LightGBM Interval Calibration

The raw prediction intervals are evaluated using coverage.

The project calculates:

```text
Coverage =
P(Actual ETA lies between Lower and Upper prediction limits)
```

A calibration factor is then tested over a range of values.

The objective is to identify a factor that produces approximately the desired **80% empirical coverage**.

The calibrated interval is therefore:

```text
Calibrated Width =
Original Width × Calibration Factor
```

This provides a practical way of correcting intervals that may initially be too narrow or too wide.

---

# 🔧 Pinball Loss

Quantile predictions are evaluated using **Pinball Loss**.

Pinball loss is particularly appropriate for quantile regression because it directly evaluates whether a model correctly estimates a specified percentile.

It is calculated for:

```text
α = 0.10
α = 0.50
α = 0.90
```

---

# 📊 Interval Analysis

The project also investigates:

* Interval width distribution
* Interval width vs distance
* Prediction interval coverage
* Calibration curves
* Calibrated interval width
* Actual vs predicted ETA
* ETA prediction with uncertainty bands

This provides insight into not only **how accurate the ETA is**, but also **how confident the model is in its prediction**.

---

# 🚀 NGBoost

NGBoost is used for probabilistic regression.

Unlike ordinary regression models that output only a point prediction, NGBoost estimates a probability distribution for the target.

A **Normal distribution** is used in the implementation.

The model produces:

```text
eta_mean
eta_std
```

where:

* `eta_mean` = predicted mean ETA
* `eta_std` = predicted standard deviation

---

# 📐 NGBoost Prediction Intervals

An approximately 80% prediction interval is constructed using:

```text
Lower = Mean − 1.28 × Standard Deviation

Upper = Mean + 1.28 × Standard Deviation
```

The resulting fields are:

```text
eta_lower
eta_upper
```

The interval can then be calibrated using a learned multiplicative factor.

---

# 📏 NGBoost Uncertainty

NGBoost uncertainty is directly represented by:

```text
eta_std
```

This enables delivery-level risk classification.

For example:

```text
Low Risk     → low predictive standard deviation
Medium Risk  → moderate predictive standard deviation
High Risk    → high predictive standard deviation
```

The exact thresholds used in the implementation can be adjusted depending on operational requirements.

---

# 📊 NGBoost Evaluation

NGBoost is evaluated using:

* MAE
* RMSE
* R²
* CRPS

---

# 📉 CRPS

The **Continuous Ranked Probability Score (CRPS)** evaluates the quality of probabilistic predictions.

Unlike a metric that evaluates only the predicted mean, CRPS considers the predicted probability distribution.

This makes it particularly useful for assessing whether the predicted uncertainty is meaningful and well-calibrated.

A Gaussian CRPS implementation is included for the NGBoost predictions.

---

# 🔬 NGBoost Explainability

Because NGBoost outputs a probability distribution rather than a simple prediction, explainability is performed using model-agnostic SHAP analysis.

The project analyzes:

### Mean ETA

Which features influence the predicted mean delivery time?

### Prediction Uncertainty

Which features influence the width of the predicted uncertainty interval?

A surrogate Gradient Boosting model is used to approximate interval width for uncertainty attribution.

---

# 📈 Partial Dependence Analysis

Partial Dependence Plots are generated for the most influential features identified through SHAP.

These plots help investigate how changes in an individual feature affect predicted ETA while averaging over the other features.

PDP analysis is performed for both:

* LightGBM-related modelling
* NGBoost-related modelling

---

# ⚠️ Risk Classification

The predicted uncertainty is converted into operational risk categories.

For LightGBM:

```text
Uncertainty =
(eta_90 - eta_10) / 2
```

For NGBoost:

```text
Uncertainty =
eta_std
```

The resulting uncertainty is categorized into:

```text
🟢 Low Risk
🟠 Medium Risk
🔴 High Risk
```

This transforms probabilistic predictions into an operationally interpretable risk signal.

---

# 🧭 Route Optimization

The project integrates ETA predictions with **Google OR-Tools** to solve a simplified Vehicle Routing Problem with Time Windows (VRPTW).

The objective is to determine a feasible sequence of delivery stops while considering:

* Delivery locations
* Travel time
* ETA-derived time windows
* Waiting/slack time
* Route duration
* Depot start/end location

---

# 🗺️ Geospatial Routing

Delivery coordinates are used to construct a travel-time matrix.

Distances are calculated using:

* Haversine distance
* Geodesic distance

Travel time is approximated using an assumed average speed of:

```text
30 km/h
```

The resulting distance matrix is converted into travel time in minutes.

---

# ⏰ ETA-Based Time Windows

The probabilistic ETA predictions are incorporated into routing.

For example, a LightGBM prediction can be transformed into:

```text
ETA = eta_50

Time Window =
eta_50 ± buffer
```

For NGBoost:

```text
Lower ETA = eta_lower
Upper ETA = eta_upper
```

and additional operational buffer is added around the predicted interval.

This means the routing algorithm does not simply find the shortest geographic route.

It attempts to find a route that is also **feasible with respect to predicted delivery timing**.

---

# 🔄 OR-Tools VRPTW

The routing model includes:

```text
Number of vehicles = 1
Depot = node 0
```

The OR-Tools routing solver uses a time dimension to enforce delivery windows.

The route is generated using a routing strategy such as:

```text
PATH_CHEAPEST_ARC
```

A feasible route is returned when the time-window constraints can be satisfied.

Example route format:

```text
Depot → Stop 4 → Stop 1 → Stop 6 → Stop 2 → Stop 3 → Stop 5 → Depot
```

---

# 🗺️ Interactive Route Maps

Folium is used to visualize the resulting routes.

The maps contain:

* Depot marker
* Delivery-stop markers
* Route path
* ETA information
* Prediction uncertainty
* Risk level

Risk levels are visually represented using different marker colors.

For example:

```text
Low Risk     → Blue
Medium Risk  → Orange
High Risk    → Red
```

The route itself is displayed using an animated path.

---

# 📍 Route Duration Comparison

The project compares route durations generated using different probabilistic ETA approaches.

For example:

```text
LightGBM Route
vs.
NGBoost Route
```

The resulting route durations can be compared through:

* Tabular summaries
* Bar charts
* Interactive maps

This provides an additional perspective on how different ETA uncertainty models affect routing decisions.

---

# 📊 Dashboard

The project is designed around an interactive **Streamlit dashboard**.

The dashboard brings together the major components of the pipeline into a single interface.

Key functionality includes:

### 🔮 ETA Prediction

Generate delivery-time predictions using the trained models.

### 📏 Uncertainty Estimation

Display:

* Predicted ETA
* Lower bound
* Upper bound
* Prediction interval
* Standard deviation

### ⚠️ Risk Analysis

Classify deliveries into:

```text
Low
Medium
High
```

risk categories.

### 🧭 Route Optimization

Generate feasible delivery routes using Google OR-Tools.

### 🌍 Geospatial Visualization

Display delivery locations and optimized routes on interactive Folium maps.

### 📊 Model Analysis

Present model performance and relevant visualizations.

---

# 🧰 Technologies Used

## Programming & Data Processing

* **Python**
* **Pandas**
* **NumPy**

## Machine Learning

* **Scikit-learn**
* **XGBoost**
* **LightGBM**
* **NGBoost**

## Optimization

* **Google OR-Tools**

Used for the Vehicle Routing Problem with Time Windows (VRPTW).

## Geospatial Analysis

* **Folium**
* **Geopy**
* Haversine/geodesic distance calculations

## Explainability

* **SHAP**

## Visualization

* **Matplotlib**
* **Seaborn**
* **Plotly**
* **Folium**

## Dashboard

* **Streamlit**

## Development / Deployment

* **Google Colab**
* **VS Code**
* **Git**
* **Docker** *(optional for deployment)*

---

# 📁 Project Structure

A recommended repository structure is:

```text
eta-prediction-optimizer/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
├── Dockerfile
│
├── data/
│   ├── delivery_data.csv
│   ├── final_dataframe.csv
│   └── updated_eta_df.csv
│
├── models/
│   ├── model_10.pkl
│   ├── model_50.pkl
│   ├── model_90.pkl
│   └── ngboost_model.pkl
│
├── notebooks/
│   └── Self Project.ipynb
│
├── maps/
│   ├── route_map_lgbm_5stops.html
│   ├── route_map_lgbm_6stops.html
│   ├── ngboost_route_with_risk_map_5stops.html
│   └── ngboost_route_with_risk_map_6stops.html
│
└── requirements.txt
```

The exact structure can be modified depending on how the final repository is organized.

---

# 📦 How to Run Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Taniya2711/eta-prediction-optimizer.git

cd eta-prediction-optimizer
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses packages including:

```text
pandas
numpy
scikit-learn
xgboost
lightgbm
ngboost
shap
folium
geopy
ortools
streamlit
matplotlib
seaborn
plotly
```

---

## 4. Run the Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The Streamlit application will then be available through the local URL displayed in the terminal.

---

# 🧪 Running the Notebook

The complete modelling workflow was developed and tested in **Google Colab**.

The notebook contains the stages for:

```text
Synthetic data generation
        ↓
Noise injection
        ↓
Data cleaning
        ↓
Feature engineering
        ↓
EDA
        ↓
Classical ML
        ↓
XGBoost
        ↓
Hyperparameter tuning
        ↓
SHAP analysis
        ↓
LightGBM quantile modelling
        ↓
NGBoost probabilistic modelling
        ↓
Calibration
        ↓
Route optimization
        ↓
Folium visualization
```

The notebook can be opened directly in Google Colab for experimentation.

---

# 🔐 Reproducibility

Random seeds are used throughout the synthetic data generation and modelling pipeline.

For example:

```python
np.random.seed(42)
random.seed(42)
```

and models use:

```text
random_state = 42
```

where applicable.

This ensures that experiments can be reproduced with consistent results.

---

# ⚙️ Key Modelling Design

One of the central design decisions in this project is the separation between:

### Point ETA Prediction

Models such as:

* Linear Regression
* Ridge
* Random Forest
* XGBoost

primarily answer:

> **"What is the predicted delivery duration?"**

### Probabilistic ETA Prediction

Models such as:

* LightGBM Quantile Regression
* NGBoost

answer:

> **"What is the predicted delivery duration, and how uncertain is that prediction?"**

This uncertainty is then used downstream by the routing component.

---

# 🔗 ML + Optimization Integration

The key contribution of the project is the connection between Machine Learning and Operations Research.

Rather than treating ETA prediction and routing as completely separate tasks, the system follows:

```text
ETA Model
   ↓
Prediction
   ↓
Prediction Uncertainty
   ↓
Delivery Time Window
   ↓
Risk Classification
   ↓
OR-Tools VRPTW
   ↓
Feasible Route
```

This creates a more complete delivery decision-support pipeline.

---

# ⚠️ Assumptions & Limitations

Because the project uses synthetic data, several assumptions are made.

### Synthetic Data

The dataset does not represent actual historical delivery operations.

Therefore, model performance should not be interpreted as production-level delivery accuracy.

### Constant Travel Speed

The routing component assumes an approximate travel speed of:

```text
30 km/h
```

Actual road networks have varying speeds based on:

* Traffic
* Road type
* Signals
* Congestion
* Road closures
* Time of day

### Simplified Geographic Distance

Distance is calculated using Haversine/geodesic calculations rather than actual road-network distances.

### Simplified Time Windows

ETA predictions are converted into operational windows using configurable buffers.

### Single Vehicle

The demonstrated OR-Tools examples primarily use:

```text
1 vehicle
```

The architecture can be extended to multiple vehicles.

### Limited Stop Demonstration

Small subsets of deliveries are used for route-feasibility demonstrations rather than optimizing all 10,000 synthetic orders simultaneously.

This keeps the demonstration computationally manageable.

---

# 🚀 Potential Future Improvements

The system can be extended significantly for production-style applications.

### Real Road-Network Routing

Replace straight-line/geodesic distances with:

* Google Maps API
* OpenStreetMap
* OSRM
* GraphHopper

to obtain actual road travel times.

### Real Traffic Data

Incorporate historical or live traffic information.

### Real Weather APIs

Use weather forecasts corresponding to the delivery location and expected delivery time.

### Dynamic Routing

Re-optimize routes when:

* New orders arrive
* Traffic changes
* Deliveries are delayed
* Vehicles become unavailable

### Multiple Vehicles

Extend the OR-Tools model to support:

* Multiple vehicles
* Vehicle capacities
* Vehicle-specific operating costs
* EV battery constraints

### Better Probabilistic Calibration

Explore:

* Conformal prediction
* Distribution-free prediction intervals
* Coverage-vs-width optimization
* Quantile calibration

### Production Deployment

The Streamlit prototype can be converted into:

```text
Frontend
   ↓
REST API
   ↓
ETA Model Service
   ↓
Optimization Service
   ↓
Database
```

and containerized using Docker.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end workflow spanning multiple areas of Data Science and Machine Learning:

### Data Science

* Synthetic data generation
* Data cleaning
* Feature engineering
* EDA
* Statistical analysis

### Machine Learning

* Linear Regression
* Ridge Regression
* Random Forest
* XGBoost

### Probabilistic Machine Learning

* LightGBM Quantile Regression
* NGBoost
* Prediction intervals
* Predictive standard deviation
* CRPS
* Pinball loss
* Calibration

### Explainable AI

* SHAP
* Feature importance
* Local explanations
* Partial Dependence
* Uncertainty attribution

### Optimization

* Vehicle Routing Problem
* Time-window constraints
* Google OR-Tools
* ETA-aware routing

### Geospatial Analytics

* K-Means location clustering
* Haversine distance
* Geodesic distance
* Interactive route visualization

### Software Engineering

* Modular modelling pipeline
* Serialized models
* CSV-based model/data interfaces
* Streamlit dashboard
* Git
* Optional Docker deployment

---

# 💡 Why This Project?

Traditional ETA systems often provide a single number:

```text
ETA = 35 minutes
```

However, two deliveries can have the same predicted ETA while having very different levels of uncertainty.

This project therefore goes one step further:

```text
Predicted ETA
      +
Prediction Uncertainty
      ↓
Risk Assessment
      ↓
Time-Window Construction
      ↓
Route Optimization
```

The result is a system that does not only ask:

> **"How long will the delivery take?"**

but also:

> **"How confident are we in that ETA, how risky is the delivery, and how should that information influence the route?"**

---

# 👩‍💻 Author

**Taniya Das**

GitHub:

```text
https://github.com/Taniya2711/eta-prediction-optimizer
```

---

# ⭐ Project Highlights

**📦 10,000 synthetic delivery orders**

**🤖 Classical + ensemble + probabilistic ML**

**📊 LightGBM quantile prediction intervals**

**🎲 NGBoost probabilistic ETA distributions**

**⚠️ ETA uncertainty-based risk classification**

**🧠 SHAP-based model explainability**

**📈 Calibration and coverage analysis**

**🧭 OR-Tools time-window route optimization**

**🌍 Interactive Folium route maps**

**🖥️ Streamlit dashboard**

**🐳 Docker-ready architecture**

---

## 📜 License

This is a self project, done as part of learning and exploration. This is not intended to be used and distributed without the author's permission.
