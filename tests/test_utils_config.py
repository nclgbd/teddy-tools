from argparse import Namespace, ArgumentParser

# teddytools imports
from teddytools.utils.config import Configuration
from teddytools.utils import repl

repl.install()


class TestConfiguration:
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

        # modules
        assert run_config.modules is not None

        modules = run_config.modules
        assert any(modules["sklearn"])
        assert any(modules["torch"])
        assert any(modules["azureml"])
