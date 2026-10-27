import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow
import mlflow.spark

# We only use utils for the Spark Session and Paths
from src.common import utils

def plot_model_comparison():
    """
    [Graph 1] Compares RMSE of the models found in your MLflow experiment.
    """
    print(">>> [1/3] Generating Model Comparison Graph...")
    
    # 1. Get Experiment
    experiment = mlflow.get_experiment_by_name("BDA_Housing_Price_Prediction")
    if not experiment:
        print("   ! Experiment not found. Please run training_pipe.py first.")
        return

    # 2. Search Runs
    df_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Filter for "AutoML" runs
    df_runs = df_runs[df_runs['tags.mlflow.runName'].str.contains("AutoML", na=False)]
    
    if df_runs.empty:
        print("   ! No AutoML runs found.")
        return

    # 3. Process Data
    df_runs['algorithm'] = df_runs['tags.mlflow.runName'].apply(lambda x: x.replace("AutoML_", ""))
    best_runs = df_runs.groupby('algorithm')['metrics.rmse'].min().reset_index()
    
    # 4. Plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(best_runs['algorithm'], best_runs['metrics.rmse'], color=['#ff9999','#66b3ff'])
    
    plt.ylabel('RMSE Error (€/m²)')
    plt.title('Model Selection: Linear Regression vs Random Forest')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{int(yval)}€", ha='center', va='bottom')

    plt.savefig("B3_model_comparison.png")
    print("   -> Saved 'B3_model_comparison.png'")
    plt.close()

def plot_residuals_and_fit(df_predictions):
    """
    [Graph 2 & 3] Accuracy & Error Analysis (Actual vs Predicted)
    """
    print(">>> [2/3] Generating Fit and Residuals Graphs...")
    
    # Convert to Pandas (limit to 2000 points for speed)
    pdf = df_predictions.select("avg_price_m2", "prediction").limit(2000).toPandas()
    pdf['residual'] = pdf['avg_price_m2'] - pdf['prediction']

    # --- Plot A: Actual vs Predicted ---
    plt.figure(figsize=(10, 6))
    plt.scatter(pdf["avg_price_m2"], pdf["prediction"], alpha=0.5, color='blue', edgecolors='k', s=30)
    
    # Ideal line
    min_val = min(pdf["avg_price_m2"].min(), pdf["prediction"].min())
    max_val = max(pdf["avg_price_m2"].max(), pdf["prediction"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
    
    plt.xlabel("Actual Price (€/m²)")
    plt.ylabel("Predicted Price (€/m²)")
    plt.title("Goodness of Fit: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.savefig("B3_actual_vs_predicted.png")
    print("   -> Saved 'B3_actual_vs_predicted.png'")
    plt.close()

    # --- Plot B: Residual Distribution ---
    plt.figure(figsize=(10, 6))
    plt.hist(pdf['residual'], bins=25, color='purple', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='r', linestyle='dashed', linewidth=2)
    plt.xlabel("Prediction Error (€/m²)")
    plt.ylabel("Frequency")
    plt.title("Residual Analysis (Error Distribution)")
    plt.savefig("B3_residuals.png")
    print("   -> Saved 'B3_residuals.png'")
    plt.close()

def plot_feature_importance(model):
    """
    [Graph 4] Explanation (Feature Importance or Coefficients)
    """
    print(">>> [3/3] Generating Feature Importance Graph...")

    # The Model is the LAST stage in our pipeline
    model_stage = model.stages[-1]
    
    importances = None
    method_name = ""

    # Check for Random Forest
    if hasattr(model_stage, "featureImportances"):
        importances = model_stage.featureImportances.toArray()
        method_name = "Feature Importance (Random Forest)"
    
    # Check for Linear Regression
    elif hasattr(model_stage, "coefficients"):
        importances = np.abs(model_stage.coefficients.toArray())
        method_name = "Absolute Coefficients (Linear Regression)"
    
    if importances is None:
        print("   ! Model does not support feature importance.")
        return

    # MAPPING FEATURES
    # We know the last 3 are: [Unemployment, Income, Immigrants]
    numeric_features = ["Unemployment Rate", "Avg Income", "Immigration Count"]
    
    numeric_importance = importances[-3:]
    neighborhood_importance = np.sum(importances[:-3]) # Sum of all neighborhood dummies
    
    labels = ["Neighborhood (Location)"] + numeric_features
    values = [neighborhood_importance] + list(numeric_importance)
    
    # Normalize to %
    total_imp = sum(values)
    values_pct = [(v / total_imp) * 100 for v in values]
    
    # Sort
    s_values, s_labels = zip(*sorted(zip(values_pct, labels)))
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(s_labels, s_values, color='green', alpha=0.6)
    
    plt.xlabel(f"Relative Impact (%) - {method_name}")
    plt.title("What drives Housing Prices in Barcelona?")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center')

    plt.savefig("B3_feature_importance.png")
    print("   -> Saved 'B3_feature_importance.png'")
    plt.close()

def main():
    spark = utils.get_spark_session("BDA_Visualization")
    spark.sparkContext.setLogLevel("ERROR")

    print("\n--- STARTING VISUALIZATION PIPELINE ---\n")

    # 1. Model Comparison (Metadata only)
    plot_model_comparison()

    # 2. Load Production Model
    print(">>> Loading 'Production' Model...")
    model_name = "HousingPricePredictor"
    model_uri = f"models:/{model_name}/Production"
    
    try:
        model = mlflow.spark.load_model(model_uri)
    except Exception:
        print(f"!!! Error: Could not load model from '{model_uri}'.")
        print("    Did you successfully run 'training_pipe.py'?")
        sys.exit(1)

    # 3. Load Data & Predict
    data_path = utils.get_path("exploitation_zone", "neighborhood_market_metrics")
    df = spark.read.format("delta").load(data_path)
    
    print(">>> Running Predictions on Data...")
    predictions = model.transform(df)

    # 4. Generate Remaining Plots
    plot_residuals_and_fit(predictions)
    plot_feature_importance(model)

    print("\n--- SUCCESS: All images saved in project root. ---")
    spark.stop()

if __name__ == "__main__":
    main()
