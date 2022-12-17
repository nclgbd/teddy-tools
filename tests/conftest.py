"""
Configurations for `pytest`.
"""

import pytest


@pytest.fixture
def run_config_yaml_path():
    return "tests/config/runconfig/run.yml"


@pytest.fixture
def model_config_yaml_path():
    # returns the path to the model configuration yaml file
    return "tests/config/runconfig/modelconfig/logisticregression.yaml"
