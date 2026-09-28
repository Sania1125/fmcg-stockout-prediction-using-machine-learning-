from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory


# ======================================================
# PATH CONFIGURATION
# ======================================================

ROOT = Path(__file__).resolve().parent

app = Flask(__name__)


MODEL_PATH = ROOT / "models" / "best_model.joblib"

META_PATH = ROOT / "models" / "model_metadata.json"

DATA_PATH = ROOT / "data" / "inventory_data.csv"

RESULTS_PATH = ROOT / "outputs" / "results" / "model_comparison.csv"

FIGURE_PATH = ROOT / "outputs" / "figures"

PREDICTION_PATH = ROOT / "outputs" / "predictions.csv"


FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================
# FEATURES USED BY MODEL
# ======================================================

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



# ======================================================
# LOAD TRAINED MODEL
# ======================================================

model = (

    joblib.load(MODEL_PATH)

    if MODEL_PATH.exists()

    else None

)



# ======================================================
# LOAD MODEL METADATA
# ======================================================

meta = (

    json.loads(
        META_PATH.read_text()
    )

    if META_PATH.exists()

    else {

        "best_model": "Not trained",

        "risk_thresholds": {

            "medium":0.35,

            "high":0.65

        }

    }

)



# ======================================================
# LOAD MODEL RESULTS
# ======================================================

results = (

    pd.read_csv(RESULTS_PATH)
    .to_dict("records")

    if RESULTS_PATH.exists()

    else []

)



# ======================================================
# LOAD DATASET
# ======================================================

def load_data():

    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

        return df


    return pd.DataFrame()



# ======================================================
# LIVE ANALYTICS GENERATOR
# ======================================================

def generate_live_analytics():


    df = load_data()


    figures = []

    stats = {}



    if df.empty:

        return figures, stats



    # ==========================
    # KPI CALCULATIONS
    # ==========================


    stats["total_records"] = len(df)



    if "current_stock" in df.columns:

        stats["average_stock"] = round(
            df["current_stock"].mean(),
            2
        )



    if "sales_velocity" in df.columns:

        stats["average_sales_velocity"] = round(
            df["sales_velocity"].mean(),
            2
        )



    if "stockout_within_7_days" in df.columns:


        stats["stockout_rate_%"] = round(

            df["stockout_within_7_days"].mean()*100,

            2

        )


        stats["total_stockouts"] = int(

            df["stockout_within_7_days"].sum()

        )




    # ==========================
    # GRAPH 1
    # CURRENT STOCK
    # ==========================


    if "current_stock" in df.columns:


        plt.figure(figsize=(8,5))


        df["current_stock"].hist()


        plt.title(
            "Current Stock Distribution"
        )


        plt.xlabel(
            "Current Stock"
        )


        plt.ylabel(
            "Frequency"
        )


        file_name = "current_stock_distribution.png"



        plt.savefig(

            FIGURE_PATH / file_name,

            bbox_inches="tight"

        )


        plt.close()



        figures.append(file_name)





    # ==========================
    # GRAPH 2
    # SALES VELOCITY
    # ==========================


    if "sales_velocity" in df.columns:


        plt.figure(figsize=(8,5))


        df["sales_velocity"].hist()


        plt.title(
            "Sales Velocity Distribution"
        )


        file_name = "sales_velocity_distribution.png"



        plt.savefig(

            FIGURE_PATH / file_name,

            bbox_inches="tight"

        )


        plt.close()



        figures.append(file_name)



    # ==========================
    # GRAPH 3
    # STOCKOUT DISTRIBUTION
    # ==========================


    if "stockout_within_7_days" in df.columns:


        plt.figure(figsize=(6,5))


        df[
            "stockout_within_7_days"
        ].value_counts().plot(
            kind="bar"
        )


        plt.title(
            "Stockout Within 7 Days"
        )


        plt.xlabel(
            "0 = No Stockout, 1 = Stockout"
        )


        file_name="stockout_distribution.png"



        plt.savefig(

            FIGURE_PATH / file_name,

            bbox_inches="tight"

        )


        plt.close()
    # ==========================
    # GRAPH 4
    # CATEGORY STOCKOUT RISK
    # ==========================


    if (
        "category" in df.columns
        and
        "stockout_within_7_days" in df.columns
    ):


        category_risk = (

            df.groupby("category")
            ["stockout_within_7_days"]
            .mean()
            .sort_values()

        )


        plt.figure(figsize=(10,5))


        category_risk.plot(
            kind="bar"
        )


        plt.title(
            "Stockout Risk by Category"
        )


        plt.ylabel(
            "Stockout Probability"
        )


        file_name="category_stockout_risk.png"


        plt.savefig(

            FIGURE_PATH / file_name,

            bbox_inches="tight"

        )


        plt.close()


        figures.append(file_name)





    # ==========================
    # GRAPH 5
    # REGION STOCKOUT RISK
    # ==========================


    if (
        "region" in df.columns
        and
        "stockout_within_7_days" in df.columns
    ):


        region_risk = (

            df.groupby("region")
            ["stockout_within_7_days"]
            .mean()
            .sort_values()

        )


        plt.figure(figsize=(10,5))


        region_risk.plot(
            kind="bar"
        )


        plt.title(
            "Stockout Risk by Region"
        )


        plt.ylabel(
            "Stockout Probability"
        )


        file_name="region_stockout_risk.png"



        plt.savefig(

            FIGURE_PATH / file_name,

            bbox_inches="tight"

        )


        plt.close()



        figures.append(file_name)



    return figures, stats





# ======================================================
# HOME PAGE
# ======================================================

@app.route("/")
def index():

    return render_template(
        "index.html",
        meta=meta
    )





# ======================================================
# PREDICTION PAGE
# ======================================================

@app.route(
    "/predict",
    methods=["GET","POST"]
)

def predict():


    prediction = None

    error = None



    if request.method == "POST":


        try:


            data = {


                "current_stock":
                float(request.form["current_stock"]),


                "sales_velocity":
                float(request.form["sales_velocity"]),


                "lead_time_days":
                float(request.form["lead_time_days"]),


                "demand_variability":
                float(request.form["demand_variability"]),


                "promotion_status":
                int(request.form["promotion_status"]),


                "price":
                float(request.form["price"]),


                "discount":
                float(request.form["discount"]),


                "category":
                request.form["category"],


                "region":
                request.form["region"],


                "weather_condition":
                request.form["weather_condition"],


                "seasonality":
                request.form["seasonality"],


                "day_of_week":
                int(request.form["day_of_week"]),


                "month":
                int(request.form["month"])

            }



            if model is None:

                raise Exception(
                    "Trained model not found."
                )



            row = pd.DataFrame(
                [data],
                columns=FEATURES
            )



            probability = float(

                model.predict_proba(row)[0,1]

            )



            medium = float(
                meta["risk_thresholds"]["medium"]
            )


            high = float(
                meta["risk_thresholds"]["high"]
            )



            if probability >= high:

                risk="High Risk"


            elif probability >= medium:

                risk="Medium Risk"


            else:

                risk="Low Risk"




            label = (

                "YES"

                if probability >= 0.50

                else "NO"

            )




            action = {


                "High Risk":
                "Review replenishment immediately.",


                "Medium Risk":
                "Monitor inventory and upcoming demand.",


                "Low Risk":
                "Continue normal inventory monitoring."

            }[risk]




            prediction = {


                "label":label,


                "probability_percent":
                round(probability*100,2),


                "risk":risk,


                "action":action,


                "inputs":data

            }





            # SAVE PREDICTION HISTORY


            save = pd.DataFrame([{

                **data,

                "probability":probability,

                "risk":risk,

                "prediction":label

            }])



            save.to_csv(

                PREDICTION_PATH,

                mode="a",

                header=not PREDICTION_PATH.exists(),

                index=False

            )




        except Exception as e:

            error=str(e)



    return render_template(

        "predict.html",

        prediction=prediction,

        error=error

    )





# ======================================================
# ANALYTICS PAGE
# ======================================================

@app.route("/analytics")
def analytics():


    figures, stats = generate_live_analytics()



    return render_template(

        "analytics.html",

        figures=figures,

        stats=stats

    )





# ======================================================
# MODEL PAGE
# ======================================================

@app.route("/models")
def models():


    return render_template(

        "models.html",

        results=results,

        meta=meta

    )





# ======================================================
# SERVE FIGURES
# ======================================================

@app.route(
    "/figures/<path:filename>"
)

def figures_file(filename):


    return send_from_directory(

        FIGURE_PATH,

        filename

    )





# ======================================================
# ABOUT
# ======================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )





# ======================================================
# RUN APPLICATION
# ======================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )


        figures.append(file_name)
