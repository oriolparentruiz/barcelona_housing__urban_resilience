#!/bin/bash
set -e

# 1. Setup: Go to Project Root (one level up from this script location)
# This assumes the script is inside /src or similar. 
# If this script is at the ROOT of your project, use: PROJECT_ROOT=$(pwd)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 2. Log Directory Setup
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# Generate a log file name
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/manual_run_collectors_${TIMESTAMP}.log"

echo ">>> Output is being saved to: $LOG_FILE"

# 3. Start Execution Block (Pipes everything to log file)
{
    echo "========================================================="
    echo "   Starting Manual Collector Batch at $(date)"
    echo "========================================================="

    # 1. Idealista
    echo "Batch Loading Idealista..."
    spark-submit src/collectors/data_collector.py \
      --source_name idealista \
      --dataset_name listings \
      --input_format json \
      --input_path "./raw_data/idealista/*.json"

    # 2. Immigrants
    echo "Batch Loading Immigrants..."
    spark-submit src/collectors/data_collector.py \
      --source_name ajbcn \
      --dataset_name immigrants \
      --input_format json \
      --input_path "./raw_data/immigrants/*.json"

    # 3. Income
    echo "Batch Loading Income..."
    spark-submit src/collectors/data_collector.py \
      --source_name ajbcn \
      --dataset_name income \
      --input_format csv \
      --delimiter "," \
      --input_path "./raw_data/income_v2/*.csv"

    # 4. Unemployment
    echo "Batch Loading Unemployment..."
    spark-submit src/collectors/data_collector.py \
      --source_name ajbcn \
      --dataset_name unemployment \
      --input_format json \
      --input_path "./raw_data/unemployment/*.json"

    echo "DONE! Data Lake Populated."

} 2>&1 | tee -a "$LOG_FILE"
