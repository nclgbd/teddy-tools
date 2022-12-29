import os

# sklearn imports
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

# local imports
from teddytools.utils.config import Configuration
from teddytools.sklearn.model import SKLearnModelConfiguration

_AVAILABLE_PREPROCESSING_STEPS = [decomposition, preprocessing]


def build_preprocessing_pipeline(
    run_config: Configuration = None,
    run_config_yaml_path: os.PathLike = "",
):
    # get the configuration
    run_config = (
        Configuration(yaml_file=run_config_yaml_path)
        if run_config is None
        else run_config
    )
    pipeline = Pipeline(steps=[])
    sklearn_configurations = run_config.sklearn
    run_type = run_config.type
    pipeline_steps = sklearn_configurations["configurations"][run_type]["steps"]

    # for step, step_params in pipeline_steps:
    for _pipeline_step in pipeline_steps:
        step_name = list(_pipeline_step.keys())[0]
        step_params = _pipeline_step[step_name]
        # console.log(f"Adding {step} to pipeline...")
        for processing_step in _AVAILABLE_PREPROCESSING_STEPS:
            if hasattr(processing_step, step_name):
                step_class = getattr(processing_step, step_name)
                step_instance = step_class(**step_params)
                pipeline.steps.append((step_name.lower(), step_instance))

    return pipeline


def build_clf(
    model_config: SKLearnModelConfiguration = None,
    model_config_yaml_path: os.PathLike = "",
    run_config: Configuration = None,
    run_config_yaml_path: os.PathLike = "",
):

    # get the configuration
    run_config = (
        Configuration(yaml_file=run_config_yaml_path)
        if run_config is None
        else run_config
    )
    model_config_yaml_path = (
        run_config.model_config_yaml_path
        if model_config_yaml_path == ""
        else model_config_yaml_path
    )

    # get the model configuration
    model_config = (
        SKLearnModelConfiguration(yaml_file=model_config_yaml_path)
        if model_config is None
        else model_config
    )

    # get the model parameters
    run_type = run_config.type if "type" in run_config else "train"
    model_config.build_model(mode=run_type)
