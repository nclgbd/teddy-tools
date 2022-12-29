"""
Configurations for `pytest`.
"""

import pytest

# sklearn imports
from sklearn.datasets import load_digits


@pytest.fixture
def data():
    return load_digits(return_X_y=True)


@pytest.fixture
def run_config_yaml_path():
    return "tests/config/runconfig/run.yml"


@pytest.fixture
def model_config_yaml_path():
    # returns the path to the model configuration yaml file
    return "tests/config/modelconfig/logisticregression.yaml"
