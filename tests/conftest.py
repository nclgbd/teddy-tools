"""
Configurations for `pytest`.
"""

import pytest

# sklearn imports
from sklearn.datasets import load_digits

# teddytools imports
from teddytools.utils.argparsers import create_default_parser


@pytest.fixture
def data():
    return load_digits(return_X_y=True)


@pytest.fixture
def parser():
    parser = create_default_parser()
    return parser


@pytest.fixture
def run_config_yaml_file():
    return "tests/config/runconfig/run.yml"


@pytest.fixture
def model_config_yaml_file():
    # returns the path to the model configuration yaml file
    return "tests/config/modelconfig/logisticregression.yaml"


@pytest.fixture
def test_artifacts_dir():
    return "tests_artifacts/"
