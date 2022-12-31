"""
`sklearn` model configuration class. Inherits from `ModelConfiguration` class to infer the model parameters from the provided configuration file.
"""
# common imports
import os

# sklearn imports
from sklearn import base, linear_model, naive_bayes, neighbors, svm, tree

# local imports
from teddytools.utils.config import Configuration, ModelConfiguration


class SKLearnModelConfiguration(ModelConfiguration):
    _SKLEARN_MODELS = [linear_model, naive_bayes, neighbors, svm, tree]

    def __init__(
        self,
        **kwargs,
    ):
        """
        Configuration class for `sklearn` models.

        ## Args:
            `parser` (`ArgumentParser`, optional): Defaults to `None`.
            `yaml_file` (`os.PathLike`, optional): Defaults to `""`.
            `**kwargs`: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model: base.BaseEstimator = self.create_model()

    def __get_base_model(self):
        for model_library in self._SKLEARN_MODELS:
            if hasattr(model_library, self.model_name):
                base_model: base.BaseEstimator = getattr(model_library, self.model_name)
                return base_model()

    def set_mode(self, mode: str = "train"):
        """
        Set the mode of the model.

        ## Args:
            `mode` (`str`): The mode of the model. Defaults to `"train"`.
        """
        self.mode = mode

    def create_model(self):
        model: base.BaseEstimator = self.__get_base_model()
        return model

    def build_model(self, mode="train", return_clf=True):
        """
        Build the model.

        ## Returns:
            `mode` (`str`): The mode of the model. Defaults to `"train"`.
            `model` (`base.BaseEstimator`): The model.
        """
        model_params = self.configurations[mode]["model_params"]
        self.set_mode(mode)
        self.model.set_params(**model_params)

        if return_clf:
            return self.model


def build_clf(
    model_config: SKLearnModelConfiguration = None,
    model_config_yaml_file: os.PathLike = "",
    run_config: Configuration = None,
    run_config_yaml_file: os.PathLike = "",
    _type: str = None,
):
    """
    Build the classifier.

    ## Args:
        `model_config` (`SKLearnModelConfiguration`, optional): . Defaults to `None`.
        `model_config_yaml_file` (`os.PathLike`, optional): Defaults to `""`.
        `run_config` (`Configuration`, optional): Defaults to `None`.
        `run_config_yaml_file` (`os.PathLike`, optional): Defaults to `""`.
        `_type` (`str`, optional): Defaults to `None`.
    """

    # get the configuration
    run_config = (
        Configuration(yaml_file=run_config_yaml_file)
        if run_config is None
        else run_config
    )
    model_config_yaml_file = (
        run_config.model_config_yaml_file
        if model_config_yaml_file == ""
        else model_config_yaml_file
    )

    # get the model configuration
    model_config = (
        SKLearnModelConfiguration(yaml_file=model_config_yaml_file)
        if model_config is None
        else model_config
    )

    # get the model parameters
    run_type = run_config.configuration if _type is None else _type
    model_config.build_model(mode=run_type)
