import os
import sys
import yaml
import logging
from pathlib import Path
from datetime import datetime
from pyspark.sql import SparkSession

# Dynamic Project Root Determination
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config():
    config_path = PROJECT_ROOT / "conf" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

CONFIG = load_config()

def get_path(zone_key, *subpaths):
    """
    Retrieves the base path from config and optionally joins subpaths.
    Usage: 
      get_path("landing_zone") 
      get_path("landing_zone", "idealista", "listings")
    """
    base_relative_path = CONFIG["paths"].get(zone_key)
    
    if not base_relative_path:
        raise ValueError(f"Zone key '{zone_key}' not found in config.yaml")

    # Combine Root + Config Path + Any additional subpaths
    full_path = PROJECT_ROOT / base_relative_path
    
    if subpaths:
        full_path = full_path.joinpath(*subpaths)
        
    return str(full_path)

def get_spark_session(app_name="DataApp"):
    packages = ",".join(CONFIG["spark"]["packages"])
    mongo_uri = os.getenv("MONGO_URI", CONFIG["mongodb"]["uri"])

    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.jars.packages", packages) \
        .config("spark.mongodb.read.connection.uri", mongo_uri) \
        .config("spark.mongodb.write.connection.uri", mongo_uri) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.default.parallelism", "8") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

def get_logger(script_name):
    """
    Returns a logger that:
    1. Prints to Console (stdout).
    2. Automatically saves to 'logs/<script_name>_<timestamp>.log'.
    """
    # 1. Define Log Directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # 2. Define Log Filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"

    # 3. Create Logger
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    # Reset handlers to avoid duplicates if run interactively
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Handler 1: File (Saves to disk)
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 5. Handler 2: Console (Prints to screen)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(message)s') # Keep console clean
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"LOGGING STARTED: Output is being saved to {log_file}")
    return logger
