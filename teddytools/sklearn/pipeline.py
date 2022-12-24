import os

# local imports
from teddytools.utils.config import Configuration
from teddytools.sklearn.model import SKLearnModelConfiguration


def build_train_pipeline(
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
        run_config["model_config_yaml_path"]
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
    run_type = run_config.type
    model_params = model_config.configurations[run_type]
    model_config.build_model()
    return run_config, model_config
