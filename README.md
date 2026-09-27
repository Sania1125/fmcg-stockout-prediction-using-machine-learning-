# FMCG Inventory Stockout Risk Prediction Using Machine Learning

A complete Flask-only university ML project for predicting whether an SKU-store-day is likely to experience a stockout within the next seven days.

## Honest dataset statement

The project uses the original downloaded file `retail_store_inventory.csv` from Kaggle: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset). Kaggle reports 73,100 rows, 15 columns, CC0 Public Domain, and describes the data as **synthetic yet realistic**. It is not claimed to be observed company data.

**Original data → derived features → derived target → ML dataset**

- Original data remains unchanged in `data/raw/`.
- Derived data is written to `data/processed/processed_inventory_data.csv`.
- The original columns include Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing and Seasonality.
- The target is not supplied by the dataset. It is derived as `1` if any of the following seven future days for the same Store ID + Product ID has `Inventory Level <= Units Sold`; otherwise `0`. This is a transparent demand-shortfall proxy, not a claimed observed stockout label.
- Historical rolling features use `shift(1)`, so no future seven-day information enters predictors. Incomplete final horizons are removed.

## Feature mapping

| Project feature | Source / derivation |
|---|---|
| Current Stock | Inventory Level |
| Sales Velocity | Prior 7-day mean of Units Sold |
| Demand Variability | Prior 7-day standard deviation of Units Sold |
| Lead Time | Source unavailable; explicit 3-day planning assumption, disclosed limitation |
| Promotion Status | Holiday/Promotion |
| Price, Discount | Price, Discount |
| Category, Region, Weather, Seasonality | Same source columns |
| Day of Week, Month | Derived from Date |

## Run in Windows PowerShell / VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train_models.py
python app.py
```

If PowerShell blocks activation, do not change system policy; run the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000/.

## Pipeline

1. Load unchanged Kaggle CSV.
2. Sort by SKU-store-day.
3. Build leakage-safe historical features.
4. Derive the seven-day binary target.
5. Use the chronological final 20% as holdout.
6. Compare Logistic Regression, KNN, Gaussian Naive Bayes, linear SVM, Decision Tree and Random Forest.
7. Calculate accuracy, precision, recall, F1, ROC-AUC and confusion-matrix counts.
8. Save the best full preprocessing/model pipeline to `models/best_model.joblib`.
9. Flask loads that saved pipeline and never retrains on page load.

## Risk thresholds

These are project-defined decision bands, not universal industry standards: Low < 35%, Medium 35–64.9%, High ≥ 65%. The result is decision support only; it is not an automatic purchase order.

## Outputs

- `outputs/results/model_comparison.csv` — actual holdout metrics
- `outputs/figures/` — EDA and model charts generated during training
- `models/best_model.joblib` — saved fitted preprocessing + model pipeline
- `models/model_metadata.json` — selected model and thresholds

## Limitations and future improvements

The Kaggle publisher labels the data synthetic; the proxy target depends on future inventory and sales; supplier lead time is unavailable and currently a disclosed three-day planning assumption; and the result is not production-ready. Future work should use observed stockout transactions, real supplier lead times, cost-sensitive threshold tuning, walk-forward validation, calibration, and monitoring for drift.
