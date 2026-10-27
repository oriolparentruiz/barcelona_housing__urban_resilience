#!/bin/bash
set -e

# Get project root dynamically
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================="
echo "   Barcelona Housing Project - Environment Setup"
echo "========================================================="

# 1. Check if venv exists
if [ -d "venv" ]; then
    echo ">>> Virtual Environment 'venv' detected."
    read -p ">>> Do you want to reinstall dependencies? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ">>> Removing old environment..."
        rm -rf venv
        REINSTALL=true
    else
        echo ">>> Skipping installation."
        REINSTALL=false
    fi
else
    REINSTALL=true
fi

# 2. Install Dependencies
if [ "$REINSTALL" = true ]; then
    echo ">>> Creating Python Virtual Environment..."
    python3 -m venv venv
    
    echo ">>> Installing dependencies (this takes ~5 mins)..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

# Ensure MongoDB is running
echo ">>> Checking MongoDB Status..."
if ! pgrep -x "mongod" > /dev/null; then
    echo "   MongoDB is stopped. Attempting to start..."
    # Try sudo start; if it fails (due to permissions), warn the user
    sudo service mongod start || echo "   [WARNING] Could not auto-start MongoDB. Please run 'sudo service mongod start' manually."
    
    # Wait a moment for it to initialize
    sleep 5
else
    echo "   MongoDB is already running."
fi

# 3. Configure Airflow
echo ">>> Configuring Local Airflow..."
export AIRFLOW_HOME="$PROJECT_ROOT/airflow_local"
mkdir -p "$AIRFLOW_HOME"

# Initialize DB (Generates the default config file)
./venv/bin/airflow db init

# --- AUTOMATED FIXES (Critical for Grading) ---
CONFIG_FILE="$AIRFLOW_HOME/airflow.cfg"

echo ">>> Patching airflow.cfg to prevent timeout errors..."

# We use python to patch the file because it is safer/cleaner than sed
./venv/bin/python - <<EOF
import os
import re

config_path = "$CONFIG_FILE"
project_dags = "$PROJECT_ROOT/dags"

with open(config_path, 'r') as f:
    content = f.read()

# Fix 1: Disable Example DAGs (Stops the 'Worker Timeout' and 'Kubernetes' errors)
content = content.replace("load_examples = True", "load_examples = False")

# Fix 2: Point to correct DAGs folder
content = re.sub(r"dags_folder = .*", f"dags_folder = {project_dags}", content)

with open(config_path, 'w') as f:
    f.write(content)
EOF
# --- END FIXES ---

# 4. Create Admin User (Silently)
echo ">>> Creating Admin User..."
./venv/bin/airflow users create \
    --username admin \
    --firstname Professor \
    --lastname User \
    --role Admin \
    --email admin@uni.edu \
    --password admin > /dev/null 2>&1 || true

echo "========================================================="
echo "   SETUP COMPLETE & PATCHED"
echo "========================================================="
echo "The configuration has been fixed automatically."
echo ""
echo "To run the project:"
echo "1. source venv/bin/activate"
echo "2. export AIRFLOW_HOME=$(pwd)/airflow_local"
echo "3. export PYTHONPATH=$(pwd)"
echo "4. airflow standalone"