# Presentation Notes — FMCG Inventory Stockout Risk Prediction

Use these 20 short slides for a 10–15 minute demonstration.

1. **Project title** — This project predicts seven-day stockout risk for SKU-store-day observations.
2. **Business problem** — Stockouts make products unavailable when customers want them and can cause lost sales.
3. **Why stockouts matter** — Managers need early warning so they can review replenishment before the next supplier cycle.
4. **Objective** — Estimate the probability of `stockout_within_7_days` from information available today.
5. **Target users** — Inventory managers, replenishment planners, store operations teams and supply-chain analysts.
6. **Original dataset** — Kaggle Retail Store Inventory Forecasting Dataset; 73,100 rows, 15 columns, CC0 Public Domain. Kaggle describes it as synthetic yet realistic.
7. **Data dictionary** — Important source fields include Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Price, Discount, Weather Condition, Holiday/Promotion and Seasonality.
8. **Feature mapping** — Inventory Level becomes current stock; Units Sold becomes historical sales velocity and variability; Holiday/Promotion becomes promotion status.
9. **Target variable** — The source does not provide the target. It is derived as 1 when future inventory is no greater than future sales on any of the next seven days for the same SKU-store history.
10. **Classification problem** — The target has two classes: 1 for risk and 0 for no derived risk.
11. **Preprocessing** — Historical rolling features use shifted sales, missing values are imputed, numeric fields are scaled, and categoricals are one-hot encoded.
12. **EDA** — The Analytics page shows stock, velocity, variability, promotion, stockout and target-distribution charts generated from actual data.
13. **Algorithms** — Logistic Regression, KNN, Gaussian Naive Bayes, calibrated linear SVM, Decision Tree and Random Forest are compared.
14. **Model comparison** — The actual CSV in `outputs/results/model_comparison.csv` contains all holdout metrics and confusion-matrix counts.
15. **Evaluation** — Accuracy alone is insufficient: recall measures missed risky cases, while precision measures false alarms.
16. **Flask application** — Flask loads `models/best_model.joblib`; it does not retrain when a page opens.
17. **Live demo** — Open `/predict`, enter values, submit, and show prediction, probability, risk band and recommendation.
18. **Business decision** — Low risk means continue monitoring; medium means review inventory and demand; high means review replenishment.
19. **Limitations** — The dataset is synthetic, the target is a transparent proxy, and supplier lead time is a disclosed three-day planning assumption because the source has no lead-time column.
20. **Conclusion** — This is a reproducible coursework decision-support prototype, not an automatic purchase-order system.

## Likely teacher questions

- **What is an SKU?** A stock-keeping unit: a distinct product identifier.
- **What is a stockout?** A situation where requested demand cannot be fulfilled because inventory is unavailable.
- **Why seven days?** It is the assignment planning horizon and creates a practical early-warning window.
- **Why classification?** The output is a binary class: stockout risk yes or no.
- **Did you create the target?** Yes. The original Kaggle file does not contain this exact seven-day label; the project derives a future inventory-versus-sales proxy.
- **How was leakage prevented?** Future observations are used only for the label. Rolling velocity and variability use `shift(1)` so they contain only prior sales.
- **Why compare multiple algorithms?** Their assumptions differ; comparison helps select a model based on actual holdout results rather than a manually chosen accuracy.
- **What does probability mean?** It is the saved model's estimated likelihood of the derived positive class for the entered feature row, not a universal guarantee.
- **How are risk bands calculated?** Project-defined thresholds: Low < 35%, Medium 35–64.9%, High ≥ 65%. They are not universal industry rules.
- **Why Flask?** It provides a lightweight Python web interface that separates saved-model inference from training.
- **What happens on submit?** Flask validates inputs, builds a one-row DataFrame with the exact trained feature schema, calls the saved pipeline, then displays the probability and recommendation.
