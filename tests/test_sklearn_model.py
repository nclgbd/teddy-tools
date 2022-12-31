import pytest

# teddytools imports
from teddytools.utils.config import Configuration, create_configuration_from_yaml
from teddytools.utils import repl
from teddytools.sklearn.model import SKLearnModelConfiguration

repl.install()


class TestSKLearnModelConfiguration:
    @pytest.fixture
    def run_config(self, run_config_yaml_file):
        run_config = create_configuration_from_yaml(
            yaml_file=run_config_yaml_file, _configuration_class=Configuration
        )
        return run_config

    @pytest.mark.smoketest
    def test_model_config_yaml_file(self, run_config, model_config_yaml_file: str):
        """
        The model configuration yaml file.

        ## Args:
            `model_config_yaml_file` (`str`): the path to the model configuration yaml file
        """
        model_config = create_configuration_from_yaml(
            yaml_file=model_config_yaml_file,
            _configuration_class=SKLearnModelConfiguration,
        )

        # model_config
        assert (
            run_config.model_config_yaml_file
            == model_config.yaml_file
            == model_config_yaml_file
        )
        assert model_config is not None
        assert model_config.model_name == "LogisticRegression"
        assert model_config.random_state == 42

        # configurations
        train_config = model_config.configurations["train"]
        assert train_config is not None
        assert train_config["model_params"]["n_jobs"] == -1
        assert train_config["train_test_split"]["test_size"] == 0.2
