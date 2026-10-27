import argparse
import sys
from pyspark.sql.functions import lit, input_file_name, regexp_extract, to_date, concat, col

# Import from the common module
from src.common import utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_name", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--input_format", required=True, choices=["json", "csv"])
    parser.add_argument("--delimiter", default=",")
    return parser.parse_args()

def main():
    # 1. Initialize Logger
    logger = utils.get_logger("data_collector")

    args = parse_args()

    # Use the shared Spark Session builder
    spark = utils.get_spark_session(f"Collector_{args.dataset_name}")

    logger.info(f"--- Batch Ingestion for {args.dataset_name} ---")

    try:
        # 1. Read Data
        logger.info(f"Reading data from: {args.input_path}")
        
        if args.input_format == "json":
            # Using multiline=true is safer for JSON arrays
            df = spark.read.option("multiline", "true").json(args.input_path)
        elif args.input_format == "csv":
            df = spark.read.option("header", "true").option("delimiter", args.delimiter).csv(args.input_path)

        # 2. Add Audit Columns
        logger.info("Adding audit columns (source_name, dataset_name, origin_file)...")
        df_enriched = df \
            .withColumn("source_name", lit(args.source_name)) \
            .withColumn("dataset_name", lit(args.dataset_name)) \
            .withColumn("origin_file", input_file_name())

        # 3. Date Extraction
        if args.dataset_name == "listings":  # Idealista logic
            df_final = df_enriched.withColumn("raw_date_str", regexp_extract("origin_file", r"(\d{4}_\d{2}_\d{2})", 1)) \
                                  .withColumn("ingestion_date", to_date(col("raw_date_str"), "yyyy_MM_dd"))
        else: # Yearly datasets logic
            df_final = df_enriched.withColumn("year_str", regexp_extract("origin_file", r"(\d{4})", 1)) \
                                  .withColumn("ingestion_date", concat(col("year_str"), lit("-01-01")))

        # 4. Write to Landing Zone
        # Uses the updated utils.get_path to join the folders automatically
        output_path = utils.get_path("landing_zone", args.source_name, args.dataset_name)

        logger.info(f"Writing data to: {output_path}")

        df_final.drop("origin_file", "raw_date_str", "year_str").write \
            .mode("overwrite") \
            .partitionBy("ingestion_date") \
            .parquet(output_path)

        logger.info(f"SUCCESS: Ingested to {output_path}")

    except Exception as e:
        logger.error(f"FAILED: {e}")
        sys.exit(1)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
