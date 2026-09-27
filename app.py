from pathlib import Path
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request, send_from_directory

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)

MODEL_PATH = ROOT / "models" / "best_model.joblib"
RESULTS_PATH = ROOT / "outputs" / "results" / "model_comparison.csv"
META_PATH = ROOT / "models" / "model_metadata.json"

FEATURES = [
    "current_stock",
    "sales_velocity",
    "lead_time_days",
    "demand_variability",
    "promotion_status",
    "price",
    "discount",
    "category",
    "region",
    "weather_condition",
    "seasonality",
    "day_of_week",
    "month"
]

# Load trained ML pipeline only.
# Training does NOT happen when the page opens.
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

results = (
    pd.read_csv(RESULTS_PATH).to_dict("records")
    if RESULTS_PATH.exists()
    else []
)

meta = (
    json.loads(META_PATH.read_text())
    if META_PATH.exists()
    else {
        "best_model": "Not trained",
        "risk_thresholds": {
            "medium": 0.35,
            "high": 0.65
        }
    }
)

figures = [
    "model_comparison.png",
    "stock_distribution.png",
    "sales_velocity.png",
    "demand_variability.png",
    "promotion_vs_stockout.png",
    "stock_vs_stockout.png",
    "target_distribution.png"
]


@app.route("/")
def index():
    return render_template(
        "index.html",
        meta=meta
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():

    # IMPORTANT:
    # When page opens, there is NO prediction.
    prediction = None
    error = None

    if request.method == "POST":

        try:
            # -----------------------------
            # GET USER INPUT
            # -----------------------------

            data = {
                "current_stock": float(
                    request.form["current_stock"]
                ),

                "sales_velocity": float(
                    request.form["sales_velocity"]
                ),

                "lead_time_days": float(
                    request.form["lead_time_days"]
                ),

                "demand_variability": float(
                    request.form["demand_variability"]
                ),

                "promotion_status": int(
                    request.form["promotion_status"]
                ),

                "price": float(
                    request.form["price"]
                ),

                "discount": float(
                    request.form["discount"]
                ),

                "category": request.form["category"],

                "region": request.form["region"],

                "weather_condition": request.form[
                    "weather_condition"
                ],

                "seasonality": request.form["seasonality"],

                "day_of_week": int(
                    request.form["day_of_week"]
                ),

                "month": int(
                    request.form["month"]
                )
            }

            # -----------------------------
            # VALIDATION
            # -----------------------------

            numeric_fields = [
                "current_stock",
                "sales_velocity",
                "lead_time_days",
                "demand_variability",
                "price",
                "discount"
            ]

            for field in numeric_fields:
                if data[field] < 0:
                    raise ValueError(
                        f"{field} cannot be negative."
                    )

            if data["day_of_week"] < 0 or data["day_of_week"] > 6:
                raise ValueError(
                    "Day of week must be between 0 and 6."
                )

            if data["month"] < 1 or data["month"] > 12:
                raise ValueError(
                    "Month must be between 1 and 12."
                )

            # -----------------------------
            # CHECK TRAINED MODEL
            # -----------------------------

            if model is None:
                raise FileNotFoundError(
                    "Trained model not found. "
                    "Run: python src/train_models.py"
                )

            # -----------------------------
            # CREATE ONE ROW FROM USER INPUT
            # -----------------------------

            row = pd.DataFrame(
                [data],
                columns=FEATURES
            )

            # -----------------------------
            # ML PREDICTION
            # -----------------------------

            probability = float(
                model.predict_proba(row)[0, 1]
            )

            label = int(
                probability >= 0.50
            )

            # -----------------------------
            # RISK
            # -----------------------------

            medium_threshold = float(
                meta["risk_thresholds"]["medium"]
            )

            high_threshold = float(
                meta["risk_thresholds"]["high"]
            )

            if probability >= high_threshold:
                risk = "High Risk"

            elif probability >= medium_threshold:
                risk = "Medium Risk"

            else:
                risk = "Low Risk"

            # -----------------------------
            # RECOMMENDATION
            # -----------------------------

            action = {
                "High Risk":
                    "Review replenishment before the next supplier cycle.",

                "Medium Risk":
                    "Review inventory and upcoming demand.",

                "Low Risk":
                    "Continue monitoring inventory."
            }[risk]

            # -----------------------------
            # RESULT
            # -----------------------------

            prediction = {
                "label": "YES" if label else "NO",
                "probability": probability,
                "probability_percent": probability * 100,
                "risk": risk,
                "action": action,
                "inputs": data
            }

            # Terminal output for checking
            print("\n========== PREDICTION ==========")
            print("User Input:")
            print(data)
            print("Probability:", probability)
            print("Risk:", risk)
            print("Stockout within 7 days:", prediction["label"])
            print("================================\n")

        except Exception as e:

            error = str(e)

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error
    )


@app.route("/models")
def models():

    return render_template(
        "models.html",
        results=results,
        meta=meta
    )


@app.route("/analytics")
def analytics():

    return render_template(
        "analytics.html",
        figures=figures
    )


@app.route("/figures/<path:filename>")
def figures_file(filename):

    return send_from_directory(
        ROOT / "outputs" / "figures",
        filename
    )


@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


if __name__ == "__main__":
    app.run(debug=True)