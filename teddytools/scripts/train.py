# common imports
import numpy as np

# sklearn imports
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
from teddytools.sklearn.model import SKLearnModelConfiguration, build_clf
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


def train(args, X: np.ndarray = None, y: np.ndarray = None):

    # step 0: load configurations
    run_config = Configuration(yaml_file=args.run_config_yaml_file)

    # enable logging via mlflow
    mlflow_enabled = run_config.use_mlflow
    if mlflow_enabled:
        mlflow_run = mlflow_setup()

    model_config_yaml_file = (
        args.model_config_yaml_file
        if "model_config_yaml_file" in args
        else run_config.model_config_yaml_file
    )
    model_config = SKLearnModelConfiguration(yaml_file=model_config_yaml_file)
    run_type = run_config.configuration
    train_config = model_config.configurations[run_type]
    random_state = run_config.random_state

    # step 1: preprocess data using preprocessing pipeline
    if run_config._test and X is None and y is None:
        X, y = load_digits(return_X_y=True)

    preprocessing_pipeline = build_pipeline(
        run_config=run_config,
    )
    X = preprocessing_pipeline.fit_transform(X, y)

    # step 2: build model
    build_clf(
        model_config=model_config,
        run_config=run_config,
    )

    # step 3: create train/test split
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
    train(args)
