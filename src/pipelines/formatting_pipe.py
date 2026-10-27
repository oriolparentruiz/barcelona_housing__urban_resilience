import argparse
import os
import sys
import traceback
from pyspark.sql.functions import (
    col, trim, when, lit, current_timestamp, explode
)
from pyspark.sql.types import IntegerType, DoubleType, BooleanType
from src.common import utils

# HELPER FUNCTION FOR IDEMPOTENCY
def write_mongo_upsert(df, collection_name, key_cols, logger):
    """
    Writes to MongoDB using Upsert logic to guarantee idempotency.
    """
    shard_key_json = "{" + ", ".join([f'"{c}": 1' for c in key_cols]) + "}"

    # Dynamic Config
    db_name = utils.CONFIG["mongodb"]["database"]

    logger.info(f"   >>> MongoDB Upsert ({collection_name}) | Key: {key_cols}")

    df.write.format("mongodb") \
        .mode("append") \
        .option("database", db_name) \
        .option("collection", collection_name) \
        .option("replaceDocument", "true") \
        .option("shardKey", shard_key_json) \
        .save()

# PROCESSING FUNCTIONS

def process_immigrants(spark, execution_date, logger):
    logger.info(f">>> Processing Immigrants for date: {execution_date}...")

    # Dynamic Path: landing_zone/ajbcn/immigrants
    input_path = utils.get_path("landing_zone", "ajbcn", "immigrants")

    if not os.path.exists(input_path):
        logger.warning(f"   [WARNING] Path does not exist locally: {input_path}")

    # Note: This reads the whole dataset. 
    # For optimization in production, consider filtering by partition if available.
    df = spark.read.parquet(input_path)

    if df.rdd.isEmpty():
        logger.warning(f">>> NO DATA found for date {execution_date}. Skipping transformation.")
        return

    # 2. TRANSFORM
    df_clean = df.select(
        col("Any").cast(IntegerType()).alias("year"),
        col("Codi_Districte").cast(IntegerType()).alias("district_code"),
        col("Nom_Districte").alias("district_name"),
        col("Codi_Barri").cast(IntegerType()).alias("neighborhood_code"),
        col("Nom_Barri").alias("neighborhood_name"),
        col("NACIONALITAT_G").cast(IntegerType()).alias("nationality_group"),
        col("EDAT_Q").cast(IntegerType()).alias("age_group"),
        col("SEXE").cast(IntegerType()).alias("sex"),
        # Handle ".." values
        when(col("Valor") == "..", None).otherwise(col("Valor")).cast(IntegerType()).alias("immigrant_count"),
        col("ingestion_date")
    )

    df_clean = df_clean.withColumn("district_name", trim(col("district_name"))) \
                       .withColumn("neighborhood_name", trim(col("neighborhood_name"))) \
                       .withColumn("formatted_timestamp", current_timestamp())

    # 3. WRITE A: MongoDB (Idempotent Upsert)
    upsert_key = ["year", "neighborhood_code", "nationality_group", "age_group", "sex", "ingestion_date"]
    write_mongo_upsert(df_clean, "immigrants", upsert_key, logger)

    # 4. WRITE B: Parquet
    fs_output_path = utils.get_path("formatted_zone", "immigrants")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")

    df_clean.write \
        .mode("overwrite") \
        .partitionBy("ingestion_date") \
        .parquet(fs_output_path)

    logger.info(f">>> SUCCESS: Immigrants processed for {execution_date}")


def process_idealista(spark, execution_date, logger):
    logger.info(f">>> Processing Idealista for date: {execution_date}...")

    input_path = utils.get_path("landing_zone", "idealista", "listings")

    if not os.path.exists(input_path):
        logger.warning(f"SKIPPING: Path not found {input_path}")
        return

    # Incremental Filter
    df = spark.read.parquet(input_path)

    if df.rdd.isEmpty():
        logger.warning(f">>> NO DATA found for date {execution_date}. Skipping.")
        return

    # Business Logic
    df_clean = df.select(
        col("propertyCode").alias("property_code"),
        col("price").cast(DoubleType()),
        col("priceByArea").cast(DoubleType()).alias("price_per_m2"),
        col("size").cast(DoubleType()),
        col("rooms").cast(IntegerType()),
        col("bathrooms").cast(IntegerType()),
        col("district").alias("district_name"),
        col("neighborhood").alias("neighborhood_name"),
        col("latitude").cast(DoubleType()),
        col("longitude").cast(DoubleType()),
        col("floor"),
        col("propertyType").alias("property_type"),
        col("hasLift").cast(BooleanType()).alias("has_lift"),
        col("exterior").cast(BooleanType()),
        col("ingestion_date")
    )

    df_clean = df_clean.withColumn("district_name", trim(col("district_name"))) \
                       .withColumn("neighborhood_name", trim(col("neighborhood_name"))) \
                       .withColumn("formatted_timestamp", current_timestamp())

    # 3. WRITE A: MongoDB
    upsert_key = ["property_code", "ingestion_date"]
    write_mongo_upsert(df_clean, "idealista", upsert_key, logger)

    # 4. WRITE B: Parquet
    fs_output_path = utils.get_path("formatted_zone", "idealista")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")

    df_clean.write \
        .mode("overwrite") \
        .partitionBy("ingestion_date") \
        .parquet(fs_output_path)

    logger.info(f">>> SUCCESS: Idealista processed for {execution_date}")


def process_unemployment(spark, execution_date, logger):
    logger.info(f">>> Processing Unemployment for date: {execution_date}...")

    input_path = utils.get_path("landing_zone", "ajbcn", "unemployment")

    if not os.path.exists(input_path):
        logger.warning(f"SKIPPING: Path not found {input_path}")
        return

    # Incremental Filter
    df = spark.read.parquet(input_path)

    if df.rdd.isEmpty():
        logger.warning(f">>> NO DATA found for date {execution_date}. Skipping.")
        return

    # Business Logic
    df_exploded = df.select(explode(col("result.records")).alias("record"), col("ingestion_date"))

    df_clean = df_exploded.select(
        col("record.Any").cast(IntegerType()).alias("year"),
        col("record.Mes").cast(IntegerType()).alias("month"),
        col("record.Codi_Districte").cast(IntegerType()).alias("district_code"),
        col("record.Nom_Districte").alias("district_name"),
        col("record.Codi_Barri").cast(IntegerType()).alias("neighborhood_code"),
        col("record.Nom_Barri").alias("neighborhood_name"),
        col("record.Nombre").cast(IntegerType()).alias("unemployment_count"),
        col("record.Demanda_ocupació").alias("demand_type"),
        when(col("record.Sexe") == "Homes", "male")
        .when(col("record.Sexe") == "Dones", "female")
        .otherwise("unknown").alias("sex"),
        col("ingestion_date")
    )

    df_clean = df_clean.withColumn("district_name", trim(col("district_name"))) \
                       .withColumn("formatted_timestamp", current_timestamp())

    # 3. WRITE A: MongoDB
    upsert_key = ["year", "month", "neighborhood_code", "demand_type", "sex", "ingestion_date"]
    write_mongo_upsert(df_clean, "unemployment", upsert_key, logger)

    # 4. WRITE B: Parquet
    fs_output_path = utils.get_path("formatted_zone", "unemployment")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")

    df_clean.write \
        .mode("overwrite") \
        .partitionBy("ingestion_date") \
        .parquet(fs_output_path)

    logger.info(f">>> SUCCESS: Unemployment processed for {execution_date}")


def process_income(spark, execution_date, logger):
    logger.info(f">>> Processing Income for date: {execution_date}...")

    input_path = utils.get_path("landing_zone", "ajbcn", "income")

    if not os.path.exists(input_path):
         logger.warning(f"   [WARNING] Python cannot find path: {input_path}")

    # Incremental Filter
    df = spark.read.parquet(input_path)

    if df.rdd.isEmpty():
        logger.warning(f">>> NO DATA found for date {execution_date}. Skipping.")
        return

    # Business Logic
    df_clean = df.select(
        col("Any").cast(IntegerType()).alias("year"),
        col("Codi_Districte").cast(IntegerType()).alias("district_code"),
        col("Nom_Districte").alias("district_name"),
        col("Codi_Barri").cast(IntegerType()).alias("neighborhood_code"),
        col("Nom_Barri").alias("neighborhood_name"),
        col("Seccio_Censal").cast(IntegerType()).alias("census_section"),
        col("Import_Euros").cast(DoubleType()).alias("amount_euros"),
        col("ingestion_date")
    )

    df_clean = df_clean.withColumn("formatted_timestamp", current_timestamp())

    # 3. WRITE A: MongoDB
    upsert_key = ["year", "neighborhood_code", "census_section", "ingestion_date"]
    write_mongo_upsert(df_clean, "income", upsert_key, logger)

    # 4. WRITE B: Parquet
    fs_output_path = utils.get_path("formatted_zone", "income")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")

    df_clean.write \
        .mode("overwrite") \
        .partitionBy("ingestion_date") \
        .parquet(fs_output_path)

    logger.info(f">>> SUCCESS: Income processed for {execution_date}")


if __name__ == "__main__":
    # 1. Initialize Logger
    logger = utils.get_logger("formatting_pipe")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True,
                        choices=["immigrants", "idealista", "unemployment", "income"],
                        help="The specific dataset to process")
    parser.add_argument("--execution_date", required=True,
                        help="The logic date of the pipeline run (YYYY-MM-DD)")
    args = parser.parse_args()

    # Initialize Spark
    spark = utils.get_spark_session(f"Fmt_{args.dataset_name}")

    # ENABLE DYNAMIC PARTITION OVERWRITE
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    try:
        logger.info(f"Starting Formatting Pipeline for {args.dataset_name} ({args.execution_date})")

        if args.dataset_name == "immigrants":
            process_immigrants(spark, args.execution_date, logger)
        elif args.dataset_name == "idealista":
            process_idealista(spark, args.execution_date, logger)
        elif args.dataset_name == "unemployment":
            process_unemployment(spark, args.execution_date, logger)
        elif args.dataset_name == "income":
            process_income(spark, args.execution_date, logger)

        logger.info(f"\n=== FORMATTING FINISHED FOR {args.dataset_name} ({args.execution_date}) ===")

    except Exception as e:
        logger.error(f"\n!!! PIPELINE FAILED: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        spark.stop()
