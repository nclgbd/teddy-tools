from argparse import Namespace, ArgumentParser
import pytest

# sklearn imports
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# teddytools imports
from teddytools.utils.config import Configuration, create_configuration_from_yaml
from teddytools.utils import repl
from teddytools.sklearn.model import SKLearnModelConfiguration, build_clf
from teddytools.sklearn.pipeline import *

repl.install()


class TestPipeline:
    @pytest.fixture
    def run_config(self, run_config_yaml_file):
        run_config = create_configuration_from_yaml(
            yaml_file=run_config_yaml_file, _configuration_class=Configuration
        )
        return run_config

    @pytest.fixture
    def model_config(self, model_config_yaml_file):
        model_config = create_configuration_from_yaml(
            yaml_file=model_config_yaml_file,
            _configuration_class=SKLearnModelConfiguration,
        )
        return model_config

    @pytest.mark.smoketest
    def test_build_preprocessing_pipeline(
        self,
        run_config: Configuration,
        data,
    ):
        """
        The run configuration yaml file.

        ## Args:
            `run_config_yaml_file` (`str`): the path to the run configuration yaml file
        """

        pipeline = build_preprocessing_pipeline(
            run_config=run_config,
        )

        assert pipeline is not None
        assert len(pipeline.steps) == 2

        X, y = data
        output = pipeline.fit_transform(X, y)

        assert output is not None

    @pytest.mark.smoketest
    def test_build_clf(
        self, run_config: Configuration, model_config: SKLearnModelConfiguration, data
    ):
        """
        The run configuration yaml file.

        ## Args:
            `run_config_yaml_file` (`str`): the path to the run configuration yaml file
        """

        preprocessing_pipeline = build_preprocessing_pipeline(
            run_config=run_config,
        )
        X, y = data
        X = preprocessing_pipeline.fit_transform(X, y)

        build_clf(
            model_config=model_config,
            run_config=run_config,
        )

        # model_config
        assert model_config.model is not None
        assert model_config.model_name == "LogisticRegression"
        assert model_config.random_state == 42

        # configurations
        train_config = model_config.configurations["train"]
        assert train_config is not None
        assert train_config["model_params"]["n_jobs"] == -1
        assert train_config["train_test_split"]["test_size"] == 0.2

        # # attempt to train with dummy data
        random_state = model_config.random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=random_state, **train_config["train_test_split"]
        )
        model = model_config.model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred is not None
        assert len(y_pred) == len(y_test)