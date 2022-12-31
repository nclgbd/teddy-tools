import os

# sklearn imports
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

# local imports
from teddytools.utils.config import Configuration, create_configuration_from_yaml

_AVAILABLE_PREPROCESSING_STEPS = [decomposition, preprocessing]


def build_preprocessing_pipeline(
    run_config: Configuration = None,
    run_config_yaml_path: os.PathLike = "",
):
    # get the configuration
    run_config = (
        create_configuration_from_yaml(
            yaml_file=run_config_yaml_path, _configuration_class=Configuration
        )
        if run_config is None
        else run_config
    )
    pipeline = Pipeline(steps=[])
    sklearn_configurations = run_config.pipelines
    pipeline_steps = sklearn_configurations["preprocessing"]["steps"]

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
