# common imports
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser

# sklearn imports
from sklearn import base
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# mlflow imports
import mlflow
from mlflow import MlflowClient

# local imports
from teddytools.utils import repl, get_console
from teddytools.utils.config import Configuration
from teddytools.utils.argparsers import create_default_parser
from teddytools.sklearn.model import SKLearnModelConfiguration
from teddytools.sklearn.pipeline import *

console = get_console()
repl.install()


def _fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


def mlflow_setup(
    uri: str = "http://localhost:5000",  # set_tracking_uri
    experiment_name: str = "sklearn",  # set_experiment
    experiment_id: str = None,  # set_experiment
    log_input_examples=False,  # autolog
    log_model_signatures=True,  # autolog
    log_models=True,  # autolog
    disable=False,  # autolog
    exclusive=False,  # autolog
    disable_for_unsupported_versions=False,  # autolog
    silent=False,  # autolog
    max_tuning_runs=5,  # autolog
    log_post_training_metrics=True,  # autolog
    registered_model_name=None,  # autolog
    pos_label=None,  # autolog
    **kwargs,
):
    mlflow.set_experiment(experiment_name=experiment_name, experiment_id=experiment_id)
    mlflow.sklearn.autolog(
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        max_tuning_runs=max_tuning_runs,
        log_post_training_metrics=log_post_training_metrics,
        registered_model_name=registered_model_name,
        pos_label=pos_label,
    )
    mlflow.set_tracking_uri(uri=uri)
    mlflow_run = mlflow.start_run()
    return mlflow_run


def main(args):

    # step 0: load configurations
    run_config_yaml_path = args.run_config_yaml_path
    run_config = Configuration(yaml_file=run_config_yaml_path)

    model_config_yaml_path = (
        args.model_config_yaml_path
        if "model_config_yaml_path" in args
        else run_config.modules["sklearn"]["model_config_yaml_path"]
    )
    model_config = SKLearnModelConfiguration(yaml_file=model_config_yaml_path)
    train_config = model_config.configurations["train"]

    # step 1a: load data
    # TODO: add way to load custom data
    if run_config._test == True:
        X, y = load_digits(return_X_y=True)

    # pre-run setup
    mlflow_enabled = run_config.use_mlflow
    if mlflow_enabled:
        mlflow_run = mlflow_setup()

    # step 1b: process data using preprocessing pipeline
    preprocessing_pipeline = build_preprocessing_pipeline(
        run_config=run_config,
    )
    X = preprocessing_pipeline.fit_transform(X, y)

    # step 2: build model
    build_clf(
        model_config=model_config,
        run_config=run_config,
    )

    # step 3: form tr
    random_state = train_config["model_params"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, **train_config["train_test_split"]
    )
    model = model_config.model

    # step 4: train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    console.print(f"accuracy: {accuracy_score(y_test, y_pred)}")

    if mlflow_enabled:
        mlflow.end_run()
        params, metrics, tags, artifacts = _fetch_logged_data(mlflow_run.info.run_id)
        console.print(f"params:\n{params}")
        console.print(f"metrics:\n{metrics}")
        console.print(f"tags:\n{tags}")
        console.print(f"artifacts:\n{artifacts}")


if __name__ == "__main__":
    parser = create_default_parser()
    args = parser.parse_args()
    main(args)
