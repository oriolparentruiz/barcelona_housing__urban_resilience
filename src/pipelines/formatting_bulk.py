import os
import sys
import traceback
from pyspark.sql.functions import (
    col, trim, when, lit, regexp_replace, current_timestamp, explode
)
from pyspark.sql.types import IntegerType, DoubleType, BooleanType

# Import your common utils
from src.common import utils

def process_immigrants(spark, logger):
    logger.info(">>> Processing Immigrants...")

    # 1. READ (From Landing)
    # Uses the new dynamic path: landing_zone/ajbcn/immigrants
    input_path = utils.get_path("landing_zone", "ajbcn", "immigrants")

    if not os.path.exists(input_path):
        logger.warning(f"   [WARNING] Python cannot find path: {input_path}")

    df = spark.read.parquet(input_path)

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
        when(col("Valor") == "..", None).otherwise(col("Valor")).cast(IntegerType()).alias("immigrant_count"),
        col("ingestion_date")
    )

    df_clean = df_clean.withColumn("district_name", trim(col("district_name"))) \
                       .withColumn("neighborhood_name", trim(col("neighborhood_name"))) \
                       .withColumn("formatted_timestamp", current_timestamp())

    # 3. WRITE A: MongoDB
    db_name = utils.CONFIG["mongodb"]["database"]
    
    logger.info(f"   Writing to MongoDB ({db_name}.immigrants)...")
    df_clean.write.format("mongodb") \
        .mode("overwrite") \
        .option("database", db_name) \
        .option("collection", "immigrants") \
        .save()

    # 4. WRITE B: Parquet (Formatted Zone)
    fs_output_path = utils.get_path("formatted_zone", "immigrants")
    
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")
    df_clean.write.mode("overwrite").parquet(fs_output_path)

    logger.info(">>> SUCCESS: Immigrants processed.")


def process_idealista(spark, logger):
    logger.info(">>> Processing Idealista...")
    
    input_path = utils.get_path("landing_zone", "idealista", "listings")

    if not os.path.exists(input_path):
        logger.warning(f"SKIPPING: Path not found {input_path}")
        return

    df = spark.read.parquet(input_path)

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

    # Write MongoDB
    db_name = utils.CONFIG["mongodb"]["database"]
    logger.info(f"   Writing to MongoDB ({db_name}.idealista)...")
    
    df_clean.write.format("mongodb") \
        .mode("overwrite") \
        .option("database", db_name) \
        .option("collection", "idealista") \
        .save()

    # Write Parquet
    fs_output_path = utils.get_path("formatted_zone", "idealista")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")
    
    df_clean.write.mode("overwrite").parquet(fs_output_path)

    logger.info(">>> SUCCESS: Idealista processed.")


def process_unemployment(spark, logger):
    logger.info(">>> Processing Unemployment...")
    
    input_path = utils.get_path("landing_zone", "ajbcn", "unemployment")

    if not os.path.exists(input_path):
        logger.warning(f"SKIPPING: Path not found {input_path}")
        return

    df = spark.read.parquet(input_path)
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

    # Write MongoDB
    db_name = utils.CONFIG["mongodb"]["database"]
    logger.info(f"   Writing to MongoDB ({db_name}.unemployment)...")

    df_clean.write.format("mongodb") \
        .mode("overwrite") \
        .option("database", db_name) \
        .option("collection", "unemployment") \
        .save()

    # Write Parquet
    fs_output_path = utils.get_path("formatted_zone", "unemployment")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")
    
    df_clean.write.mode("overwrite").parquet(fs_output_path)

    logger.info(">>> SUCCESS: Unemployment processed.")


def process_income(spark, logger):
    logger.info(">>> Processing Income...")

    input_path = utils.get_path("landing_zone", "ajbcn", "income")

    if not os.path.exists(input_path):
         logger.warning(f"   [WARNING] Python cannot find path: {input_path}")

    df = spark.read.parquet(input_path)

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

    # Write MongoDB
    db_name = utils.CONFIG["mongodb"]["database"]
    logger.info(f"   Writing to MongoDB ({db_name}.income)...")

    df_clean.write.format("mongodb") \
        .mode("overwrite") \
        .option("database", db_name) \
        .option("collection", "income") \
        .save()

    # Write Parquet
    fs_output_path = utils.get_path("formatted_zone", "income")
    logger.info(f"   Writing to Formatted Zone: {fs_output_path}")
    
    df_clean.write.mode("overwrite").parquet(fs_output_path)

    logger.info(">>> SUCCESS: Income processed.")


if __name__ == "__main__":
    # 1. Initialize Logger
    logger = utils.get_logger("formatting_bulk")
    
    # 2. Initialize Spark
    spark = utils.get_spark_session("BDA_P2_Formatting_Bulk")
    
    try:
        process_immigrants(spark, logger)
        process_idealista(spark, logger)
        process_unemployment(spark, logger)
        process_income(spark, logger)
        
        logger.info("\n=== FORMATTING PIPELINE FINISHED SUCCESSFULLY ===")
    
    except Exception as e:
        logger.error(f"\n!!! PIPELINE FAILED: {e}")
        # Detailed traceback to help debug
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        spark.stop()
