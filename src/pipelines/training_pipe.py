import os
import sys
import time
import traceback
import warnings

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import col

from src.common import utils

# Silence specific deprecation warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # 1. Initialize Logger
    logger = utils.get_logger("training_pipe")

    # 2. Initialize Spark
    spark = utils.get_spark_session("BDA_P2_Training_Optimized")
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # ====================================================
        # 1. Load Data
        # ====================================================
        input_path = utils.get_path("exploitation_zone", "neighborhood_market_metrics")

        if not os.path.exists(input_path):
            logger.error(f"ERROR: Data not found at {input_path}")
            sys.exit(1)

        logger.info(f">>> Reading Delta Table from: {input_path}")
        df = spark.read.format("delta").load(input_path)

        if df.count() == 0:
            logger.error("ERROR: Dataset is empty.")
            sys.exit(1)

        # ====================================================
        # PART 1: Feature Engineering (Stabilized)
        # ====================================================
        logger.info(">>> Formatting and Standardizing data...")

        # Define stages
        indexer = StringIndexer(inputCol="official_name", outputCol="neighborhood_idx", handleInvalid="keep")
        encoder = OneHotEncoder(inputCols=["neighborhood_idx"], outputCols=["neighborhood_vec"])

        feature_cols = ["neighborhood_vec", "avg_monthly_unemployed", "avg_income", "total_immigrants"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip")

        # CRITICAL: StandardScaler prevents L-BFGS failures by normalizing scales
        # (e.g. Income=30,000 vs Unemployment=0.15)
        scaler = StandardScaler(inputCol="features_raw", outputCol="scaled_features", withStd=True, withMean=True)

        # ====================================================
        # PART 2: Splitting & Caching
        # ====================================================
        train_data_raw, test_data = df.randomSplit([0.8, 0.2], seed=42)

        # Filter out extreme outliers that might confuse the linear solver
        train_data = train_data_raw.filter(col("avg_price_m2") < 6500)

        # Cache to speed up the iterative training
        train_data.cache()
        test_data.cache()

        logger.info(f"Training Rows (Cleaned):  {train_data.count()}")
        logger.info(f"Test Rows (Evaluation):   {test_data.count()}")

        # ====================================================
        # PART 3: AutoML Loop (Optimized)
        # ====================================================
        experiment_name = "BDA_Housing_Price_Prediction"
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow Experiment set to: {experiment_name}")

        evaluator_rmse = RegressionEvaluator(labelCol="avg_price_m2", predictionCol="prediction", metricName="rmse")

        # --- Model 1: Linear Regression (Stabilized) ---
        # SOLVER CHANGE: "auto" lets Spark choose the best solver.
        # If regParam > 0, it often avoids the fragile L-BFGS paths.
        lr = LinearRegression(featuresCol="scaled_features", labelCol="avg_price_m2", solver="auto")
        lr_pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, lr])

        # GRID CHANGE: Removed 0.0. Using pure 0.0 (OLS) on collinear data crashes the math.
        # We enforce at least a tiny bit of regularization (0.01).
        lr_grid = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5]) \
            .build()

        # --- Model 2: Random Forest (Robust) ---
        rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="avg_price_m2", seed=42)
        rf_pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, rf])
        rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5, 10]).build()

        candidates = [
            {"name": "LinearRegression", "pipeline": lr_pipeline, "grid": lr_grid},
            {"name": "RandomForest", "pipeline": rf_pipeline, "grid": rf_grid}
        ]

        best_global_rmse = float("inf")
        best_global_run_id = None
        best_global_name = None

        logger.info("\n>>> Starting Optimized AutoML Loop...")

        for candidate in candidates:
            model_name = candidate["name"]
            logger.info(f"\n   Optimizing {model_name}...")

            with mlflow.start_run(run_name=f"AutoML_{model_name}") as run:
                cv = CrossValidator(estimator=candidate["pipeline"],
                                    estimatorParamMaps=candidate["grid"],
                                    evaluator=evaluator_rmse,
                                    numFolds=2) # Keep folds low for speed

                try:
                    cvModel = cv.fit(train_data)
                    best_model = cvModel.bestModel
                    predictions = best_model.transform(test_data)

                    if predictions.count() > 0:
                        rmse = evaluator_rmse.evaluate(predictions)
                        r2 = RegressionEvaluator(labelCol="avg_price_m2", metricName="r2").evaluate(predictions)
                    else:
                        rmse = float('inf')
                        r2 = 0.0

                    logger.info(f"      [RESULT] {model_name} -> RMSE: {rmse:.2f} | R2: {r2:.4f}")

                    mlflow.log_metric("rmse", rmse)
                    mlflow.log_metric("r2", r2)

                    # Log model using the modern "spark" flavor
                    mlflow.spark.log_model(best_model, "model")

                    if rmse < best_global_rmse:
                        best_global_rmse = rmse
                        best_global_run_id = run.info.run_id
                        best_global_name = model_name

                except Exception as e:
                    logger.error(f"      [ERROR] Training failed for {model_name}: {e}")
                    logger.error(traceback.format_exc())

        # ====================================================
        # PART 4: Deployment (Modernized)
        # ====================================================
        logger.info("\n" + "="*50)
        logger.info(f"GLOBAL WINNER: {best_global_name} (RMSE: {best_global_rmse:.4f})")
        logger.info("="*50)

        model_registry_name = "HousingPricePredictor"
        client = MlflowClient()
        production_rmse = float('inf')

        # 1. Get Current Production RMSE (Using Modern MLflow API)
        try:
            # Search specifically for the model with the alias or tag "Production"
            versions = client.search_model_versions(f"name='{model_registry_name}'")
            prod_versions = [v for v in versions if v.current_stage == "Production"]

            if prod_versions:
                latest_prod = prod_versions[0] # Get latest production version
                prod_run = client.get_run(latest_prod.run_id)
                if "rmse" in prod_run.data.metrics:
                    production_rmse = prod_run.data.metrics["rmse"]
                    logger.info(f"Current Production RMSE: {production_rmse:.4f}")
                else:
                    logger.warning("Production model has no RMSE metric logged. Assuming infinite.")
            else:
                logger.info("No Production model found. This will be the first.")

        except Exception as e:
            logger.warning(f"Metrics check skipped or failed: {e}")

        # 2. Compare & Promote
        if best_global_rmse < production_rmse:
            logger.info(">>> RESULT: New model is BETTER. Promoting to Production.")

            # Register Model
            result = mlflow.register_model(
                model_uri=f"runs:/{best_global_run_id}/model",
                name=model_registry_name
            )
            time.sleep(2)

            # Transition to Production
            client.transition_model_version_stage(
                name=model_registry_name,
                version=result.version,
                stage="Production",
                archive_existing_versions=True
            )
            logger.info(f">>> SUCCESS: Version {result.version} is LIVE.")
        else:
            logger.info(f">>> RESULT: New model ({best_global_rmse:.2f}) is WORSE than Production ({production_rmse:.2f}). Discarding.")

        train_data.unpersist()
        test_data.unpersist()

    except Exception as e:
        logger.error(f"\n!!! PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
