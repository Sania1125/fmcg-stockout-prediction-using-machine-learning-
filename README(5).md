# FMCG Inventory Stockout Risk Prediction Using Machine Learning

A Flask-based machine learning application that estimates whether an **SKU-store-day observation is likely to experience a stockout within the next seven days**. The system converts historical inventory and sales information into leakage-safe features, compares multiple classification algorithms, saves the selected model, and provides a web interface for risk prediction and decision support.

> **Important:** This project uses the Kaggle **Retail Store Inventory Forecasting Dataset**, which Kaggle describes as **synthetic yet realistic**. It is not presented as real company transaction data. The seven-day stockout target is also a project-derived proxy rather than an observed stockout label.

---

## 1. Project Overview

### Business Problem

Stockouts occur when a product is unavailable when customers want to purchase it. For inventory teams, an early warning can help them review stock levels, demand patterns, promotions, and replenishment needs before the next planning cycle.

This project addresses the following business question:

> **Can historical inventory, sales, demand variability, pricing, promotion, and store/product information be used to estimate the risk of a stockout during the next seven days?**

### Project Objective

The main objective is to build a **binary classification system** that estimates:

- **0 — No derived stockout risk**
- **1 — Derived stockout risk within 7 days**

The application also converts the model probability into project-defined risk bands:

- **Low Risk:** probability < 35%
- **Medium Risk:** 35%–64.9%
- **High Risk:** ≥ 65%

These thresholds are **project decision bands**, not universal industry standards.

---

## 2. Target Users and Business Decision

### Target Users

- Inventory managers
- Replenishment planners
- Store operations teams
- Supply-chain analysts

### Decision Supported by the Prediction

The prediction is intended as **decision support**, not an automatic purchase-order system.

| Risk Level | Suggested Business Action |
|---|---|
| Low | Continue normal monitoring |
| Medium | Review inventory, recent demand, and upcoming promotions |
| High | Review replenishment requirements and investigate stockout risk promptly |

---

## 3. Dataset

The project uses the original:

**Retail Store Inventory Forecasting Dataset**

Source: Kaggle — Retail Store Inventory Forecasting Dataset

The dataset contains approximately **73,100 rows and 15 columns**. Kaggle describes it as synthetic but realistic and lists it as CC0 Public Domain.

### Important Source Fields

- Date
- Store ID
- Product ID
- Category
- Region
- Inventory Level
- Units Sold
- Units Ordered
- Demand Forecast
- Price
- Discount
- Weather Condition
- Holiday/Promotion
- Competitor Pricing
- Seasonality

The original CSV is preserved unchanged in:

```text
data/raw/retail_store_inventory.csv
```

---

## 4. Target Variable Creation

The original dataset does **not** contain the exact target required by this project.

The project derives:

```text
stockout_within_7_days
```

The target is assigned as **1** when, for the same Store ID + Product ID, any of the following seven future days satisfies:

```text
Inventory Level <= Units Sold
```

Otherwise, the target is **0**.

Therefore, this is a **future demand-shortfall proxy**, not a directly observed stockout label.

Only the future observations required to construct the target are used for labeling. They are not used as predictor values.

---

## 5. Feature Engineering

The model uses the following features:

| Project Feature | Source / Derivation |
|---|---|
| Current Stock | Inventory Level |
| Sales Velocity | Prior 7-day mean of Units Sold |
| Demand Variability | Prior 7-day standard deviation of Units Sold |
| Lead Time | 3-day planning assumption because source data has no lead-time field |
| Promotion Status | Holiday/Promotion |
| Price | Price |
| Discount | Discount |
| Category | Category |
| Region | Region |
| Weather Condition | Weather Condition |
| Seasonality | Seasonality |
| Day of Week | Derived from Date |
| Month | Derived from Date |

### Leakage Prevention

Historical rolling features use shifted values (`shift(1)`), so the model's predictor values are based only on information available before the prediction point.

The incomplete final seven-day horizons are removed when constructing the target.

---

## 6. Machine Learning Workflow

The project follows this pipeline:

```text
Original Kaggle Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Seven-Day Target Creation
        ↓
Chronological Train/Test Split
        ↓
Preprocessing
        ↓
Multiple Model Training
        ↓
Holdout Evaluation
        ↓
Model Comparison
        ↓
Selected Model Saved
        ↓
Flask Web Application
        ↓
Stockout Risk Prediction
```

### Preprocessing

**Numerical features**

- Missing-value imputation using median
- Standard scaling

**Categorical features**

- Missing-value imputation using most frequent value
- One-hot encoding
- Unknown categories are handled safely

The preprocessing and model are stored together in a scikit-learn pipeline.

---

## 7. Models Compared

The project compares six classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes
4. Linear Support Vector Machine (SVM)
5. Decision Tree
6. Random Forest

The models are evaluated on the chronological final **20% holdout set**.

### Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion-matrix counts: TN, FP, FN, TP

For stockout-risk classification, **precision, recall, F1-score, and ROC-AUC should be considered alongside accuracy**, because a high accuracy value can be misleading when the positive class is relatively uncommon.

---

## 8. Actual Model Comparison

The following results are taken from:

```text
outputs/results/model_comparison.csv
```

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 49.59% | 3.73% | 51.41% | 6.96% | 0.511 |
| Decision Tree | 37.91% | 3.60% | 61.77% | 6.80% | 0.488 |
| SVM | 96.33% | 0.00% | 0.00% | 0.00% | 0.512 |
| KNN | 96.33% | 0.00% | 0.00% | 0.00% | 0.508 |
| Naive Bayes | 96.33% | 0.00% | 0.00% | 0.00% | 0.505 |
| Random Forest | 96.22% | 0.00% | 0.00% | 0.00% | 0.488 |

### Selected Model

The training pipeline selects the model by sorting the holdout results by:

1. **F1-score**
2. **Recall**
3. **ROC-AUC**

The current project selected:

**Logistic Regression**

The fitted pipeline is saved as:

```text
models/best_model.joblib
```

### Important Interpretation of the Current Results

The current dataset produces a strong class imbalance on the holdout set. Several models achieve approximately **96% accuracy while predicting zero positive cases**, which results in **0% precision, recall, and F1-score**.

For this reason, the project does **not** treat accuracy alone as evidence of a good stockout-risk model.

The current Logistic Regression model has:

- Accuracy: **49.59%**
- Precision: **3.73%**
- Recall: **51.41%**
- F1-score: **6.96%**
- ROC-AUC: **0.511**

This means the current model should be presented as a **coursework prototype and decision-support demonstration**, not as a production-ready stockout prediction system.

---

## 9. Web Application

The project uses **Flask** rather than Streamlit.

The Flask application:

1. Loads the saved model pipeline.
2. Accepts prediction inputs through a web form.
3. Builds a DataFrame using the trained feature schema.
4. Generates a binary risk prediction.
5. Calculates the model's positive-class probability.
6. Converts the probability into a project-defined risk band.
7. Displays a recommendation for inventory review.

The application **does not retrain the model when the page is opened**.

### Main Pages

- Home
- Prediction Workbench
- Model Comparison
- Analytics
- About

---

## 10. Project Structure

```text
fmcg-stockout-risk-prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── PRESENTATION_NOTES.md
│
├── data/
│   ├── raw/
│   │   └── retail_store_inventory.csv
│   └── processed/
│       └── processed_inventory_data.csv
│
├── models/
│   ├── best_model.joblib
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   │   ├── model_comparison.png
│   │   ├── stock_distribution.png
│   │   ├── sales_velocity.png
│   │   ├── demand_variability.png
│   │   ├── promotion_vs_stockout.png
│   │   ├── stock_vs_stockout.png
│   │   └── target_distribution.png
│   │
│   └── results/
│       └── model_comparison.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── train_models.py
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── models.html
│   ├── analytics.html
│   └── about.html
│
└── static/
    └── style.css
```

---

## 11. Installation and Setup

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Train and compare the models

```powershell
python src\train_models.py
```

This generates:

- Model comparison results
- Model comparison chart
- EDA charts
- `best_model.joblib`
- `model_metadata.json`

### 6. Start the Flask application

```powershell
python app.py
```

### 7. Open the application

Open:

```text
http://127.0.0.1:5000/
```

---

## 12. Outputs

### Model Results

```text
outputs/results/model_comparison.csv
```

Contains the actual holdout metrics and confusion-matrix counts.

### Trained Model

```text
models/best_model.joblib
```

Contains the fitted preprocessing and selected classification model.

### Model Metadata

```text
models/model_metadata.json
```

Contains:

- Selected model
- Feature list
- Target name
- Risk thresholds

### Analytics

The project generates visualizations for:

- Current stock distribution
- Sales velocity
- Demand variability
- Promotion vs. derived stockout
- Stock vs. derived stockout
- Target distribution
- Model comparison

---

## 13. Why Multiple Models?

Different algorithms make different assumptions about the relationship between the input features and the target.

Comparing several models allows the project to make the model-selection process measurable instead of choosing an algorithm arbitrarily.

The final model is selected from the actual chronological holdout results using F1-score, recall, and ROC-AUC rather than accuracy alone.

---

## 14. Limitations

This project has several important limitations:

1. The source dataset is **synthetic**, not observed company data.
2. The target is a **derived stockout proxy**, not a directly recorded stockout event.
3. Supplier lead time is not available in the source dataset, so the project uses a disclosed **3-day planning assumption**.
4. The current holdout results show substantial class-imbalance effects.
5. The model should not be used to automatically place purchase orders.
6. The current results are not sufficient to claim production-level predictive performance.

---

## 15. Future Improvements

A production-oriented version could improve the project by using:

- Real observed stockout transaction data
- Actual supplier lead times
- Supplier reliability information
- Reorder points and safety-stock levels
- Cost-sensitive learning
- Threshold tuning based on business costs
- Probability calibration
- Walk-forward / time-series validation
- Class-imbalance strategies
- Model monitoring and drift detection
- Production data pipelines
- Automated model retraining

---

## 16. Conclusion

This project demonstrates a complete machine learning workflow for **FMCG inventory stockout-risk decision support**.

It covers:

- Business problem definition
- Dataset preparation
- Feature engineering
- Leakage-aware target construction
- Classification
- Multiple-model comparison
- Holdout evaluation
- Model persistence
- Flask deployment
- Risk interpretation
- Business-oriented recommendations

The system is best described as a **reproducible university/coursework prototype for early stockout-risk analysis**, with clear limitations and opportunities for improvement.

---

## 17. Key Viva / Presentation Statement

> **“Our project predicts the probability of a derived seven-day stockout risk for an SKU-store-day observation using historical inventory, sales velocity, demand variability, promotion, pricing, and contextual features. We compare six classification models on a chronological holdout set and select the model using F1-score, recall, and ROC-AUC rather than relying on accuracy alone. The selected model is deployed through a Flask web application for decision support.”**
