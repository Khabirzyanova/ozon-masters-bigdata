from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.bash import BashSensor
from airflow.sensors.filesystem import FileSensor
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from datetime import datetime

base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
TRAIN_PATH = "/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json"
TRAIN_PATH_OUT = "user/Khabirzyanova/Khabirzyanova_train_out.parquet"
TEST_PATH = "/datasets/amazon/all_reviews_5_core_test_extra_small_features.json"
TEST_PATH_OUT = "user/Khabirzyanova/Khabirzyanova_test_out.parquet"

dsenv = "/opt/conda/envs/dsenv/bin/python"

dag = DAG(
  "Khabirzyanova_dag",
  schedule_interval=None,
  catchup=False,
  start_date = datetime(2022, 5, 29)
)

  
feature_eng_train_task = SparkSubmitOperator(
  task_id = "feature_eng_train_task",
  dag=dag,
  application=f"{base_dir}/Khabirzyanova_preprocessing.py",
  application_args=["--path-in", TRAIN_PATH, "--path-out", TRAIN_PATH_OUT ],
  spark_binary = "/usr/bin/spark-submit",
  env_vars={"PYSPARK_PYTHON": dsenv}
)

download_train_task = BashOperator(
  task_id = "download_train_task",
  dag=dag,
  bash_command = f"hdfs dfs -copyToLocal {TRAIN_PATH_OUT} {base_dir}/Khabirzyanova_train_out_local.parquet"
)

train_task = BashOperator(
  task_id = "train_task",
  dag=dag,
  bash_command = f"{dsenv} {base_dir}/Khabirzyanova_train.py --train-in {base_dir}/Khabirzyanova_train_out_local.parquet --sklearn-model-out {base_dir}/6.joblib"
  )

model_sensor = FileSensor(
  task_id = "model_sensor", 
  dag=dag,
  filepath = f'{base_dir}/6.joblib'
)


feature_eng_test_task = SparkSubmitOperator(
  task_id = "feature_eng_test_task",
  dag=dag,
  application=f"{base_dir}/Khabirzyanova_preprocessing.py",
  application_args=["--path-in", TEST_PATH, "--path-out", TEST_PATH_OUT ],
  spark_binary = "/usr/bin/spark-submit",
  env_vars={"PYSPARK_PYTHON": dsenv}
)


PREDICTION_PATH = f'Khabirzyanova_hw6_prediction'
predict_task = SparkSubmitOperator(
  task_id = "predict_task",
  dag=dag,
  application=f"{base_dir}/Khabirzyanova_prediction.py",
  application_args=["--test-in ", TEST_PATH_OUT, "--pred-out", PREDICTION_PATH, "--sklearn-model-in", f"{base_dir}/6.joblib"],
  spark_binary = "/usr/bin/spark-submit",
  env_vars={"PYSPARK_PYTHON": dsenv}
)

feature_eng_train_task >> download_train_task >> train_task >> model_sensor >> feature_eng_test_task >> predict_task
