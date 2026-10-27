import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# ==============================================================================
# 1. DYNAMIC CONFIGURATION & LOGGING SETUP
# ==============================================================================
DAGS_FOLDER = Path(__file__).resolve().parent
PROJECT_ROOT = DAGS_FOLDER.parent
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
PYTHON_EXEC = sys.executable

env_config = os.environ.copy()
env_config["PYTHONPATH"] = str(PROJECT_ROOT)

def get_logged_command(task_name, python_cmd):
    """
    Wraps the command to save stdout/stderr to a log file 
    inside the PROJECT_ROOT/logs directory.
    """
    log_file = f"{LOGS_DIR}/{task_name}_{{{{ ds }}}}.log"
    return f"""
    mkdir -p {LOGS_DIR} && \
    echo "--- Starting Task: {task_name} at $(date) ---" >> {log_file} && \
    {python_cmd} >> {log_file} 2>&1 && \
    echo "--- Task Finished Successfully ---" >> {log_file}
    """

# ==============================================================================
# 2. DAG DEFINITION
# ==============================================================================
default_args = {
    'owner': 'student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'bda_urban_resilience_pipeline',
    default_args=default_args,
    description='End-to-end Data Lake Pipeline with Local Logging',
    schedule_interval='@daily',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['bda', 'spark', 'mlops']
) as dag:

    start = EmptyOperator(task_id='pipeline_start')
    end = EmptyOperator(task_id='pipeline_end')

    # --- ZONE 1: INGESTION ---
    # We define the raw command first, then wrap it in the logger.

    cmd_idealista = f"""
    {PYTHON_EXEC} src/collectors/data_collector.py \
        --source_name idealista \
        --dataset_name listings \
        --input_format json \
        --input_path "raw_data/idealista/*.json"
    """
    ingest_idealista = BashOperator(
        task_id='ingest_idealista',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('ingest_idealista', cmd_idealista)
    )

    cmd_immigrants = f"""
    {PYTHON_EXEC} src/collectors/data_collector.py \
        --source_name ajbcn \
        --dataset_name immigrants \
        --input_format json \
        --input_path "raw_data/immigrants/*.json"
    """
    ingest_immigrants = BashOperator(
        task_id='ingest_immigrants',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('ingest_immigrants', cmd_immigrants)
    )

    cmd_unemployment = f"""
    {PYTHON_EXEC} src/collectors/data_collector.py \
        --source_name ajbcn \
        --dataset_name unemployment \
        --input_format json \
        --input_path "raw_data/unemployment/*.json"
    """
    ingest_unemployment = BashOperator(
        task_id='ingest_unemployment',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('ingest_unemployment', cmd_unemployment)
    )

    cmd_income = f"""
    {PYTHON_EXEC} src/collectors/data_collector.py \
        --source_name ajbcn \
        --dataset_name income \
        --input_format csv \
        --delimiter "," \
        --input_path "raw_data/income_v2/*.csv"
    """
    ingest_income = BashOperator(
        task_id='ingest_income',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('ingest_income', cmd_income)
    )

    # --- ZONE 2: FORMATTING ---

    cmd_fmt_idealista = f"""
    {PYTHON_EXEC} src/pipelines/formatting_pipe.py \
        --dataset_name idealista \
        --execution_date {{{{ ds }}}}
    """
    format_idealista = BashOperator(
        task_id='format_idealista',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('format_idealista', cmd_fmt_idealista)
    )

    cmd_fmt_immigrants = f"""
    {PYTHON_EXEC} src/pipelines/formatting_pipe.py \
        --dataset_name immigrants \
        --execution_date {{{{ ds }}}}
    """
    format_immigrants = BashOperator(
        task_id='format_immigrants',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('format_immigrants', cmd_fmt_immigrants)
    )

    cmd_fmt_unemployment = f"""
    {PYTHON_EXEC} src/pipelines/formatting_pipe.py \
        --dataset_name unemployment \
        --execution_date {{{{ ds }}}}
    """
    format_unemployment = BashOperator(
        task_id='format_unemployment',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('format_unemployment', cmd_fmt_unemployment)
    )

    cmd_fmt_income = f"""
    {PYTHON_EXEC} src/pipelines/formatting_pipe.py \
        --dataset_name income \
        --execution_date {{{{ ds }}}}
    """
    format_income = BashOperator(
        task_id='format_income',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('format_income', cmd_fmt_income)
    )

    # --- ZONE 3: EXPLOITATION ---
    cmd_exploit = f"{PYTHON_EXEC} src/pipelines/exploitation_pipe.py"
    exploit_zone = BashOperator(
        task_id='exploit_zone',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('exploit_zone', cmd_exploit)
    )

    # --- ZONE 4: TRAINING ---
    cmd_train = f"{PYTHON_EXEC} src/pipelines/training_pipe.py"
    train_model = BashOperator(
        task_id='train_model',
        cwd=str(PROJECT_ROOT),
        env=env_config,
        bash_command=get_logged_command('train_model', cmd_train)
    )

    # --- DEPENDENCIES ---
    # 1. Start triggers all Ingestions
    start >> [ingest_idealista, ingest_immigrants, ingest_unemployment, ingest_income]

    # 2. Swimlanes: Ingest -> Format (Parallel tracks)
    ingest_idealista >> format_idealista
    ingest_immigrants >> format_immigrants
    ingest_unemployment >> format_unemployment
    ingest_income >> format_income

    # 3. Synchronize: All formats must finish before Exploitation
    [format_idealista, format_immigrants, format_unemployment, format_income] >> exploit_zone

    # 4. Sequence: Exploit -> Train -> End
    exploit_zone >> train_model >> end
