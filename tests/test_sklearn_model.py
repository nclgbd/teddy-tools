from argparse import Namespace, ArgumentParser
import pytest

# teddytools imports
from teddytools.utils.config import Configuration
from teddytools.utils import repl
from teddytools.sklearn.model import SKLearnModelConfiguration

repl.install()


class TestSKLearnModelConfiguration:
    @pytest.fixture
    def run_config(self, run_config_yaml_path):
        run_config = Configuration(yaml_file=run_config_yaml_path)
        return run_config

    def test_model_config_yaml_path_from_conftest(self, model_config_yaml_path: str):
        """
        The model configuration yaml file.

        ## Args:
            `model_config_yaml_path` (`str`): the path to the model configuration yaml file
        """
        model_config = SKLearnModelConfiguration(yaml_file=model_config_yaml_path)

        # model_config
        assert model_config.model is not None
        assert model_config.model_name == "LogisticRegression"
        assert model_config.random_state == 42

        # configurations
        train_config = model_config.configurations["train"]
        assert train_config is not None
        assert train_config["model_params"]["random_state"] == 42
        assert train_config["train_test_split"]["test_size"] == 0.2

    def test_model_config_yaml_path_from_run_yml(self, run_config: Configuration):
        """
        The model configuration yaml file.

        ## Args:
            `model_config_yaml_path` (`str`): the path to the model configuration yaml file
        """
        sklearn_run_config = run_config.modules["sklearn"]
        model_config_yaml_path = sklearn_run_config["model_config_yaml_path"]
        model_config = SKLearnModelConfiguration(yaml_file=model_config_yaml_path)

        # model_config
        assert model_config.model is not None
        assert model_config.model_name == "LogisticRegression"
        assert model_config.random_state == 42

        # configurations
        train_config = model_config.configurations["train"]
        assert train_config is not None
        assert train_config["model_params"]["random_state"] == 42
        assert train_config["train_test_split"]["test_size"] == 0.2
