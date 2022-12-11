from argparse import Namespace, ArgumentParser
import pytest

# sklearn
from teddytools.utils import repl
from teddytools.sklearn.models import SKLearnModelConfiguration

repl.install()

@pytest.fixture
def model_config_yaml_file():
    return "tests/config/modelconfig/logisticregression.yaml"


class TestModels:
    def test_model_config_yaml_file(self, model_config_yaml_file):

        model_config = SKLearnModelConfiguration(yaml_file=model_config_yaml_file)
        assert model_config.model is not None
        assert model_config.model_name == "LogisticRegression"
        assert model_config.model_params["random_state"] == 42

    # def test_model_config_parser(self, model_config_yaml_file):
    #     parser = ArgumentParser()
    #     parser.add_argument("--model_config_yaml_file", type=str, default=model_config_yaml_file)
    #     model_config = SKLearnModelConfiguration(parser=parser)
    #     assert model_config.model is not None
    #     assert model_config.model_name == "LogisticRegression"
    #     assert model_config.model_params["random_state"] == 42
