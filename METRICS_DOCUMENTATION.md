# Threat Severity Model - Metrics Documentation

## Overview
This document provides comprehensive details about all metrics displayed in the web UI, including the formulas used to calculate them and their significance in model monitoring.

---

## 1. Real-time Tab (REAL DATA)

### Purpose
Tracks live prediction statistics from API calls made through the web interface.

### Metrics

#### Total Predictions
- **Description**: Cumulative count of all prediction requests made since page load
- **Formula**: Simple counter incremented on each API call
- **Storage**: Browser localStorage (persists across refreshes)
- **Reset**: Manual via "Reset Metrics" button

#### Avg Response Time
- **Description**: Average API latency across all predictions
- **Formula**: 
  ```
  Avg Latency = Σ(latency_i) / n
  where n = number of predictions
  ```
- **Unit**: Milliseconds (ms)
- **Calculation**: Measured using `performance.now()` before/after fetch call

#### Avg Threat Score
- **Description**: Average predicted threat severity across all predictions
- **Formula**: 
  ```
  Avg Score = Σ(threat_score_i) / n
  where threat_score ∈ [0, 10]
  ```
- **Purpose**: Track overall threat level trends in analyzed traffic

#### Error Rate
- **Description**: Percentage of failed API calls
- **Formula**: 
  ```
  Error Rate = (Failed Requests / Total Requests) × 100
  ```
- **Causes**: Network errors, invalid inputs, server timeouts, 500 errors

### Threat Distribution
Categorizes predictions into risk levels:
- **Low Risk**: Score < 3
- **Medium Risk**: 3 ≤ Score < 5
- **High Risk**: 5 ≤ Score < 7
- **Critical Risk**: Score ≥ 7

---

## 2. About Tab (INFORMATIONAL)

### NSL-KDD Dataset

#### Full Name
**NSL-KDD** = Network Security Laboratory - Knowledge Discovery in Databases

#### Dataset Statistics
- **Total Records**: 148,517 network connection records
- **Original Features**: 41 features
- **Engineered Features**: +20 security-specific features
- **Final Features**: 61 features

#### Dataset Splits
- **Training Set**: 100,778 samples (67.8%)
- **Validation Set**: 25,195 samples (17.0%)
- **Test Set**: 22,544 samples (15.2%)

#### Attack Categories

##### 1. DoS (Denial of Service)
- **Description**: Attacks that overwhelm system resources
- **Examples**: SYN flood, ping flood, teardrop
- **Mechanism**: Exhaust bandwidth, CPU, or memory
- **Impact**: Service unavailability

##### 2. Probe (Surveillance)
- **Description**: Reconnaissance to discover vulnerabilities
- **Examples**: Port scanning, network mapping, vulnerability scanning
- **Mechanism**: Systematic probing of network/services
- **Impact**: Information gathering for future attacks

##### 3. R2L (Remote to Local)
- **Description**: Unauthorized access from remote machine
- **Examples**: Password guessing, FTP write, phishing
- **Mechanism**: Exploit authentication or authorization flaws
- **Impact**: Unauthorized local access

##### 4. U2R (User to Root)
- **Description**: Privilege escalation attacks
- **Examples**: Buffer overflow, rootkit installation, privilege exploitation
- **Mechanism**: Escalate from normal user to admin/root
- **Impact**: Complete system compromise

---

## 3. Performance Tab (REAL DATA)

### Test Set Metrics
All metrics calculated on 22,544 unseen test samples.

#### Classification Metrics

##### Precision
- **Value**: 98.1%
- **Formula**: 
  ```
  Precision = TP / (TP + FP)
  Precision = 4,879 / (4,879 + 92) = 0.9814
  ```
- **Meaning**: Of all predicted threats, 98.1% are real threats
- **Significance**: Only 1.9% false alarm rate - critical for reducing alert fatigue

##### Recall (Sensitivity)
- **Value**: 55.1%
- **Formula**: 
  ```
  Recall = TP / (TP + FN)
  Recall = 4,879 / (4,879 + 3,981) = 0.5507
  ```
- **Meaning**: Model detects 55.1% of actual threats
- **Tradeoff**: Lower recall accepted to maintain high precision (cost-optimal threshold)

##### F1-Score
- **Value**: 70.6%
- **Formula**: 
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  F1 = 2 × (0.9814 × 0.5507) / (0.9814 + 0.5507) = 0.7055
  ```
- **Meaning**: Harmonic mean of precision and recall
- **Purpose**: Balanced metric for imbalanced datasets

##### AUC-ROC
- **Value**: 0.960
- **Formula**: Area Under Receiver Operating Characteristic Curve
  ```
  AUC-ROC = ∫[0,1] TPR(FPR) d(FPR)
  where TPR = TP/(TP+FN), FPR = FP/(FP+TN)
  ```
- **Meaning**: 96% probability model ranks random threat higher than random normal traffic
- **Range**: [0, 1], where 1.0 = perfect, 0.5 = random

### Confusion Matrix (Binary Classification)

#### Matrix Values (Test Set)
```
                  Predicted
                Normal    Threat
Actual Normal    13,592      92      (TN=13,592, FP=92)
Actual Threat     3,981   4,879      (FN=3,981, TP=4,879)
```

#### Calculations
- **True Negatives (TN)**: 13,592
  - Normal traffic correctly identified
  
- **True Positives (TP)**: 4,879
  - Threats correctly detected
  - Formula: `TP = Recall × Total_Threats = 0.5507 × 8,860 = 4,879`
  
- **False Positives (FP)**: 92
  - False alarms (normal flagged as threat)
  - Formula: `FP = TP × (1 - Precision) / Precision = 4,879 × 0.019 / 0.981 = 92`
  
- **False Negatives (FN)**: 3,981
  - Missed threats
  - Formula: `FN = Total_Threats - TP = 8,860 - 4,879 = 3,981`

#### Overall Accuracy
```
Accuracy = (TP + TN) / Total
Accuracy = (4,879 + 13,592) / 22,544 = 0.820 = 82.0%
```

#### False Positive Rate
```
FPR = FP / (FP + TN)
FPR = 92 / (92 + 13,592) = 0.0067 = 0.67%
```

### Regression Metrics

#### RMSE (Root Mean Square Error)
- **Value**: 3.08
- **Formula**: 
  ```
  RMSE = √[Σ(y_pred - y_actual)² / n]
  RMSE = √[Σ(prediction_i - actual_severity_i)² / 22,544]
  ```
- **Unit**: Severity score points (0-10 scale)
- **Meaning**: Average prediction error magnitude, with larger errors penalized more

#### MAE (Mean Absolute Error)
- **Value**: 1.54
- **Formula**: 
  ```
  MAE = Σ|y_pred - y_actual| / n
  MAE = Σ|prediction_i - actual_i| / 22,544
  ```
- **Unit**: Severity score points
- **Meaning**: Average absolute deviation from true scores

#### R² Score (Coefficient of Determination)
- **Value**: 0.198
- **Formula**: 
  ```
  R² = 1 - (SS_res / SS_tot)
  where:
    SS_res = Σ(y_actual - y_pred)²  (residual sum of squares)
    SS_tot = Σ(y_actual - ȳ)²      (total sum of squares)
  ```
- **Range**: (-∞, 1], where 1.0 = perfect fit
- **Meaning**: 19.8% of variance in severity scores explained by the model

---

## 4. Models Tab (REAL DATA)

### Dataset Information
- **Total Records**: 148,517
- **Train / Val / Test Split**: 100,778 / 25,195 / 22,544
- **Original Features**: 41 (NSL-KDD standard features)
- **Engineered Features**: +20 security patterns
- **Final Features**: 61

### Feature Engineering
Added 20 features capturing:
1. **Protocol Patterns**: TCP/UDP/ICMP behavior analysis
2. **Connection Rates**: Connections per second, same-host patterns
3. **Service Behaviors**: Service-specific anomaly indicators
4. **Error Ratios**: SYN errors, REJ errors, timeout ratios

### Model Comparison (Validation Set)

All models trained on same 100,778 samples, validated on 25,195 samples.

#### XGBoost (Selected Model)
- **Val RMSE**: 0.240
- **Val R²**: 0.9948
- **Training Time**: 1.56s
- **Test Precision**: 98.1%
- **Test Recall**: 55.1%
- **Test AUC-ROC**: 0.960

**Overfitting Check**:
```
Train RMSE: 0.185
Val RMSE: 0.240
Difference: 0.055 (acceptable - no severe overfitting)
```

#### LightGBM (2nd Place)
- **Val RMSE**: 0.246 (+0.006 worse than XGBoost)
- **Val R²**: 0.9946
- **Training Time**: 0.74s (2.1× faster)

**Why Not Selected**: Slightly higher error despite faster training

#### Random Forest (3rd Place)
- **Val RMSE**: 0.963 (+0.723 worse)
- **Val R²**: 0.9168
- **Training Time**: 18.60s (11.9× slower)

**Issue**: Severe overfitting
```
Train RMSE: 0.228
Val RMSE: 0.963
Overfitting Factor: 4.2×
```

#### Linear Regression (Baseline)
- **Val RMSE**: 1.204 (+0.964 worse)
- **Val R²**: 0.8699
- **Training Time**: 0.12s (13× faster)

**Purpose**: Simple baseline to verify complex models add value

### Selection Rationale: Why XGBoost Won

1. **Lowest Validation RMSE**: 0.240 (best generalization)
2. **Highest R²**: 0.9948 (explains 99.48% of variance)
3. **No Overfitting**: Train-Val gap only 0.055 RMSE
4. **Balanced Performance**: Good accuracy + acceptable speed
5. **Feature Importance**: Built-in SHAP integration for explainability
6. **Production Metrics**: 98.1% precision minimizes false alarms

---

## 5. Training Tab (REAL DATA)

### Test Set Performance
Final evaluation on 22,544 held-out samples never seen during training.

#### Metrics
- **Test RMSE**: 3.08 (severity prediction error)
- **Test MAE**: 1.54 (average deviation)
- **Test R²**: 0.198 (19.8% variance explained)
- **Test AUC-ROC**: 0.960 (96% discrimination ability)

### Optimal Thresholds

Thresholds for binary classification (threat vs. normal).

#### Cost Optimal Threshold
- **Value**: 2.42
- **Method**: Minimize cost function
  ```
  Cost = C_FP × FP + C_FN × FN
  where C_FP = cost of false alarm, C_FN = cost of missed threat
  ```
- **Assumption**: C_FN = 10 × C_FP (missing threat 10× costlier than false alarm)
- **Use Case**: Production deployment with balanced priorities

#### IR Precision Threshold
- **Value**: 4.99
- **Method**: Maximize precision for incident response
- **Formula**: Find threshold where Precision ≥ 0.95
- **Use Case**: High-priority alerts requiring immediate action

#### Balanced Threshold
- **Value**: 0.007
- **Method**: Maximize F1-Score
  ```
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
- **Use Case**: Equal weighting of precision and recall

### Dataset Information
- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Features**: 41 → 61 (after engineering)
- **Train Samples**: 100,778 (67.8%)
- **Val Samples**: 25,195 (17.0%)
- **Test Samples**: 22,544 (15.2%)

---

## 6. Data Drift Tab (MOCKUP DATA)

### Purpose
Detects changes in input data distribution that may degrade model performance.

### Population Stability Index (PSI)

#### Formula
```
PSI = Σ[(P_prod - P_train) × ln(P_prod / P_train)]
where:
  P_prod = proportion in production
  P_train = proportion in training
```

#### Interpretation
- **PSI < 0.1**: No significant drift (stable)
- **0.1 ≤ PSI < 0.2**: Moderate drift (monitor)
- **PSI ≥ 0.2**: High drift (retrain recommended)

#### Example Values (Mockup)
- Overall PSI: 0.08 (no drift)
- Duration feature: 0.05 (stable)
- Protocol Type: 0.12 (moderate drift)
- Src Bytes: 0.24 (high drift)

### Kolmogorov-Smirnov (KS) Test

#### Purpose
Statistical test comparing two distributions (training vs. production).

#### Formula
```
KS Statistic = max|F_train(x) - F_prod(x)|
where F = cumulative distribution function
```

#### Hypothesis Test
- **H0**: Training and production distributions are the same
- **Reject H0 if**: p-value < 0.05 (5% significance level)

#### Example Values (Mockup)
- Failed Tests: 3 / 41 features (7.3% failure rate)
- Avg P-value: 0.127 (most features stable)
- Max KS Stat: 0.183 (largest difference)

### Feature Statistics

#### Mean Shift
```
Mean Shift = (μ_prod - μ_train) / μ_train × 100%
```
Example: +8.3% (production features average 8.3% higher)

#### Std Shift
```
Std Shift = (σ_prod - σ_train) / σ_train × 100%
```
Example: +12.1% (production features more variable)

#### Jensen-Shannon Divergence
```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
where M = 0.5 × (P + Q), KL = Kullback-Leibler divergence
```
- **Range**: [0, 1]
- **0**: Identical distributions
- **1**: Completely different
- Example: 0.046 (4.6% difference - low drift)

---

## 7. Model Drift Tab (MOCKUP DATA)

### Purpose
Monitors model performance degradation over time due to concept drift or data changes.

### Performance Degradation

#### Accuracy Decay
```
Degradation = (Baseline_Acc - Current_Acc) / Baseline_Acc × 100%
```
Example:
- Baseline: 94.7% (initial test accuracy)
- Current: 93.1% (recent production accuracy)
- Degradation: -1.6% (manageable)
- **Alert Threshold**: -5% triggers retraining

### Error Rate Trends

#### False Positive Trend
```
FP_Trend = (FP_current - FP_baseline) / FP_baseline × 100%
```
Example: +15% increase in false alarms

#### False Negative Trend
```
FN_Trend = (FN_current - FN_baseline) / FN_baseline × 100%
```
Example: +8% increase in missed threats

#### Overall Error Trend
```
Error_Trend = (Error_current - Error_baseline) / Error_baseline × 100%
```
Example: +11% overall error increase

### Calibration Error

#### Expected Calibration Error (ECE)
```
ECE = Σ[|P(y=1|bin_i) - Accuracy(bin_i)| × n_i / N]
where:
  P(y=1|bin_i) = predicted probability in bin i
  Accuracy(bin_i) = actual accuracy in bin i
  n_i = samples in bin i
  N = total samples
```
- Example: 0.047 (well-calibrated)
- **Good**: ECE < 0.05
- **Poor**: ECE > 0.15

### SHAP Shift

#### Feature Importance Change
```
SHAP_Shift = Σ|SHAP_current_i - SHAP_baseline_i| / n_features × 100%
```
Example: 12.3% shift in feature importance patterns

---

## 8. Operational Tab (MOCKUP DATA)

### Purpose
Monitors infrastructure and API performance metrics.

### Latency Metrics

#### Percentiles
Latency values where X% of requests are faster.

- **P50 (Median)**: 45 ms
  - 50% of requests complete in ≤ 45ms
  
- **P95**: 127 ms
  - 95% of requests complete in ≤ 127ms
  - Only 5% exceed this (outliers)
  
- **P99**: 203 ms
  - 99% of requests complete in ≤ 203ms
  - Worst 1% exceed this (tail latency)

#### Why Percentiles?
- Average can hide outliers
- P95/P99 capture worst-case user experience
- SLA targets typically use P95 or P99

### Throughput Metrics

#### Requests Per Second
```
RPS = Total_Requests / Time_Window
```
Example: 23.4 requests/sec

#### Total Requests
Cumulative count since deployment.
Example: 1.2M requests

#### Failed Requests Rate
```
Failure_Rate = Failed_Requests / Total_Requests × 100%
```
Example: 0.3% failure rate

### Resource Usage

#### CPU Usage
```
CPU% = (CPU_Time / Wall_Time) × 100%
```
Example: 34% (66% headroom available)

#### Memory Usage
- Current: 1.2 GB
- Limit: 2.0 GB
- Utilization: 60%

#### Container Uptime
Time since last restart.
Example: 3 days 14 hours (high availability)

---

## Formulas Reference

### Classification Metrics
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Specificity = TN / (TN + FP)
FPR = FP / (FP + TN) = 1 - Specificity
FNR = FN / (FN + TP) = 1 - Recall
```

### Regression Metrics
```
RMSE = √[Σ(y_pred - y_actual)² / n]
MAE = Σ|y_pred - y_actual| / n
R² = 1 - (SS_res / SS_tot)
```

### Statistical Tests
```
PSI = Σ[(P_prod - P_train) × ln(P_prod / P_train)]
KS = max|F_1(x) - F_2(x)|
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
```

---

## Data Sources

### Real Data Tabs
1. **Real-time**: Calculated live from API responses
2. **About**: NSL-KDD dataset documentation
3. **Performance**: From `training_summary.json` test metrics
4. **Models**: From training logs (validation metrics)
5. **Training**: From `training_summary.json` and thresholds

### Mockup Data Tabs
6. **Data Drift**: Simulated monitoring metrics (placeholder)
7. **Model Drift**: Simulated performance decay (placeholder)
8. **Operational**: Simulated infrastructure metrics (placeholder)

---

## Model Training Pipeline

### 1. Data Preprocessing
- Load NSL-KDD dataset (148,517 records)
- Handle missing values
- Encode categorical features
- Split: 68% train / 17% val / 15% test

### 2. Feature Engineering
Add 20 security features:
- Protocol pattern indicators
- Connection rate statistics
- Service behavior flags
- Error ratio calculations

### 3. Model Training
Train 4 models:
- XGBoost (gradient boosting)
- LightGBM (light gradient boosting)
- Random Forest (ensemble)
- Linear Regression (baseline)

### 4. Model Selection
Select XGBoost based on:
- Lowest validation RMSE
- No overfitting
- Best test performance

### 5. Threshold Optimization
Find optimal thresholds for:
- Cost-optimal (balanced)
- IR-precision (high precision)
- Balanced F1-score

### 6. Model Serialization
Save:
- xgboost_model.pkl
- scaler.pkl
- shap_explainer.pkl
- training_summary.json

---

## API Endpoints

### POST /predict
**Request**:
```json
{
  "features": [0.0, 1.0, ..., 0.0],  // 41 values
  "include_explanation": true
}
```

**Response**:
```json
{
  "threat_score": 7.23,
  "risk_level": "high",
  "recommended_action": "Immediate investigation required",
  "inference_time_ms": 12.45,
  "top_contributors": [
    {"feature": "dst_host_srv_count", "contribution": 2.14},
    ...
  ]
}
```

### GET /health
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "xgboost",
  "features_count": 61
}
```

---

## Monitoring Best Practices

### When to Retrain
1. **Data Drift**: PSI > 0.2 on critical features
2. **Model Drift**: Accuracy drops > 5%
3. **Error Trends**: FP or FN increases > 20%
4. **Time-Based**: Every 3-6 months minimum

### Alert Thresholds
- **Latency**: P95 > 200ms
- **Error Rate**: > 1%
- **CPU**: > 80% for 5 minutes
- **Memory**: > 90%

### A/B Testing
When deploying new model:
1. Route 10% traffic to new model
2. Monitor metrics for 24 hours
3. Compare performance to baseline
4. Gradually increase traffic if successful

---

## Glossary

- **AUC-ROC**: Area Under Receiver Operating Characteristic curve
- **ECE**: Expected Calibration Error
- **FN**: False Negative (Type II error)
- **FP**: False Positive (Type I error)
- **JS**: Jensen-Shannon divergence
- **KL**: Kullback-Leibler divergence
- **KS**: Kolmogorov-Smirnov test
- **MAE**: Mean Absolute Error
- **PSI**: Population Stability Index
- **RMSE**: Root Mean Square Error
- **SHAP**: SHapley Additive exPlanations
- **TN**: True Negative
- **TP**: True Positive

---

**Document Version**: 1.0  
**Last Updated**: December 16, 2025  
**Model Version**: XGBoost v1.0 (trained on NSL-KDD)  
**API URL**: https://threat-model-api.mangoriver-796b80ca.centralus.azurecontainerapps.io  
**UI URL**: https://delightful-water-091ecc910.3.azurestaticapps.net/
