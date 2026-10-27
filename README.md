# BDA Project: Barcelona Housing & Urban Resilience

## 1. Project Overview
This project implements a full Data Engineering and MLOps lifecycle to analyze housing prices in Barcelona relative to urban indicators (immigration, unemployment, income). The architecture is fully reproducible and decoupled from the host file system.

### Architecture Layers
1.  **Orchestration:** Apache Airflow manages dependencies and execution.
2.  **Ingestion (Landing Zone):** Batch ingestion of JSON/CSV data into Parquet.
3.  **Processing (Formatted Zone):** Spark cleaning, standardization, and loading into **MongoDB** (Idempotent Upsert).
4.  **Exploitation (Exploitation Zone):** Aggregation into a **Delta Lake** table for analysis.
5.  **Machine Learning:** Spark ML pipelines with **MLflow** tracking and Model Registry deployment.

---

## 2. Prerequisites
Ensure the following are installed on your Linux machine before running:

* **Linux OS** (Ubuntu/Debian recommended)
* **Python 3.10+** (venv module required)
* **Java 8 or 11** (Required for PySpark execution)
* **MongoDB** installed and running on `localhost:27017`

---

## 3. Project Structure
The code is organized to separate configuration, orchestration, and logic:

```text
.
├── conf/
│   └── config.yaml          # Central configuration (paths, spark settings)
├── dags/
│   └── main_orchestrator.py # Airflow DAG definition
├── src/
│   ├── collectors/          # Ingestion logic (Spark)
│   ├── pipelines/           # Transformation & ML logic (Spark)
│   └── common/              # Shared utilities (Config loader, Spark Session)
├── scripts/                 # Setup and execution helpers
└── raw_data/                # Input datasets (JSON/CSV)
```

## 4. Quick Start Guide

**Note for Evaluators:**
The project uses a local virtual environment to ensure reproducibility and prevent conflicts with your system libraries.

### Step 1: Environment Setup (One-time)

Run the automated setup script. This creates a Python virtual environment, installs all 180+ dependencies (Spark, Airflow, MLflow), and initializes the local Airflow database.

*If you have already run this once, the script will offer to skip re-installation.*

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Step 2: Activate Environment
```bash
source venv/bin/activate
```

### Step 3: Run the Orchestrator (Airflow)
We execute Airflow in "Standalone" mode. We must export environment variables so Airflow detects the local project structure.
```bash
# 1. Point Airflow to the local folder (not user home)
export AIRFLOW_HOME=$(pwd)/airflow_local

# 2. Add the current folder to Python Path so Airflow finds 'src' modules
export PYTHONPATH=$(pwd)

# 3. Start the server
airflow standalone
```
Access the UI:

- URL: http://localhost:8080

- User: admin

- Password: admin

To Run the Pipeline: In the Airflow UI, find the DAG named bda_urban_resilience_pipeline. Click the Unpause toggle (ON) and then the Trigger DAG (Play) button.

## 5. Monitoring & ML Artifacts
### **Inspecting Model Runs (MLflow)**
The training pipeline logs metrics (RMSE, MAE) and model artifacts to a local MLflow tracking server. To view them:

1. Open a new terminal.

2. Navigate to the project root and activate the environment:
```bash
source venv/bin/activate
```
3. Launch the UI
```bash
mlflow ui --port 5000
```
4. Visit http://localhost:5000.

## 6. Configuration
All hardcoded paths have been removed. Configuration is managed in conf/config.yaml.

- Paths: Relative to the project root.

- Spark: Configures memory and Delta Lake jars.

- MongoDB: Connection strings.

To change the MongoDB connection string or folder names, edit conf/config.yaml directly.

