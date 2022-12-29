from argparse import Namespace, ArgumentParser
import pytest

# sklearn imports
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# teddytools imports
from teddytools.utils.config import Configuration
from teddytools.utils import repl
from teddytools.sklearn.model import SKLearnModelConfiguration
from teddytools.sklearn.pipeline import *
from teddytools.sklearn.train import main

repl.install()


class TestTrain:

    def test_basic_train(self, run_config_yaml_path, model_config_yaml_path):
        args = Namespace(
            run_config_yaml_path=run_config_yaml_path,
            model_config_yaml_path=model_config_yaml_path,
        )

        main(args)

        assert True