import os
from argparse import ArgumentParser

from sklearn import base, linear_model, naive_bayes, neighbors, svm, tree

# local imports
from teddytools.utils.config import ModelConfiguration


class SKLearnModelConfiguration(ModelConfiguration):
    _SKLEARN_MODELS = [linear_model, naive_bayes, neighbors, svm, tree]

    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        """
        Configuration class for `sklearn` models.

        ## Args:
            `parser` (`ArgumentParser`, optional): Defaults to `None`.
            `yaml_file` (`os.PathLike`, optional): Defaults to `""`.
        """
        super().__init__(parser, yaml_file, **kwargs)
        self.model: base.BaseEstimator = self.create_model()
        # if self.mode == "train":
        #     self.model.set_params(**self.model_params)

    def __get_base_model(self):
        for attr in self._SKLEARN_MODELS:
            if hasattr(attr, self.model_name):
                base_model: base.BaseEstimator = getattr(attr, self.model_name)
                return base_model

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
