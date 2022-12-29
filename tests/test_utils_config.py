import pytest
from argparse import Namespace, ArgumentParser

# teddytools imports
from teddytools.utils.config import Configuration, ModelConfiguration
from teddytools.utils import repl

repl.install()


class TestConfiguration:
    @pytest.mark.smoketest
    def test_run_config_yaml_path(self, run_config_yaml_path: str):
        """
        The run configuration yaml file.

        ## Args:
            `run_config_yaml_path` (`str`): the path to the run configuration yaml file
        """
        run_config = Configuration(yaml_file=run_config_yaml_path)

        # run_config
        assert run_config is not None
        assert run_config.type == "train"
        assert any(run_config.sklearn)
        assert any(run_config.torch)
        assert any(run_config.azureml)

    @pytest.mark.smoketest
    def test_model_config_yaml_path(self, model_config_yaml_path: str):
        """
        The model configuration yaml file.

        ## Args:
            `model_config_yaml_path` (`str`): the path to the model configuration yaml file
        """
        model_config = ModelConfiguration(yaml_file=model_config_yaml_path)

        # model_config
        assert model_config is not None
        assert any(model_config.configurations["train"])
