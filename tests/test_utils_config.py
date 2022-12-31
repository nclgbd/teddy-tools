import pytest
from argparse import Namespace, ArgumentParser

# teddytools imports
from teddytools.utils.config import (
    create_configuration_from_yaml,
    Configuration,
    ModelConfiguration,
)
from teddytools.utils import repl

repl.install()


class TestConfiguration:
    @pytest.mark.smoketest
    def test_create_run_configuration_from_yaml(self, run_config_yaml_file: str):
        """
        The run configuration yaml file.

        ## Args:
            `run_config_yaml_file` (`str`): the path to the run configuration yaml file
        """
        run_config = create_configuration_from_yaml(
            yaml_file=run_config_yaml_file, _configuration_class=Configuration
        )

        # run_config checks
        assert run_config is not None
        assert run_config._test == True

        # run_config.pipeline checks
        run_pipelines = run_config.pipelines
        assert any(run_pipelines)

    # @pytest.mark.smoketest
    # def test_run_config_parser(self, parser: ArgumentParser):
    #     """
    #     The run configuration yaml file.

    #     ## Args:
    #         `run_config_yaml_file` (`str`): the path to the run configuration yaml file
    #     """
    #     run_config = Configuration(parser=parser)

    #     # run_config checks
    #     assert run_config is not None
    #     assert run_config._test == True

    #     # run_config.pipeline checks
    #     run_pipelines = run_config.pipelines
    #     assert any(run_pipelines)

    @pytest.mark.smoketest
    def test_create_model_configuration_from_yaml(self, model_config_yaml_file: str):
        """
        The model configuration yaml file.

        ## Args:
            `model_config_yaml_file` (`str`): the path to the model configuration yaml file
        """
        model_config = create_configuration_from_yaml(
            yaml_file=model_config_yaml_file, _configuration_class=ModelConfiguration
        )

        # model_config
        assert model_config is not None
        assert any(model_config.configurations["train"])
        assert any(model_config.configurations["train"]["model_params"])
