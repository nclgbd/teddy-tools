from argparse import Namespace
import pytest

# teddytools imports
from teddytools.utils import repl, get_console
from teddytools.train import train

repl.install()
console = get_console()


class TestTrain:
    def test_basic_train(self, run_config_yaml_file, model_config_yaml_file):
        args = Namespace(
            run_config_yaml_file=run_config_yaml_file,
            model_config_yaml_file=model_config_yaml_file,
        )

        train(args)

        assert True
