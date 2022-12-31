"""
Configuration class for ML related tasks. Meant to be built off of for downstream tasks.
"""

# common imports
from argparse import Namespace, ArgumentParser
import yaml
import os


class _BaseConfiguration(Namespace):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        """
        Base configuration class. Built off of the `Namespace` class in `argparse` module.

        ## Args:
            `parser` (`ArgumentParser`, optional): Defaults to `None`.
            `yaml_file` (`os.PathLike`, optional): Defaults to `""`.
            `**kwargs`: Additional keyword arguments.
        """
        self.__args: dict = {}
        if parser is not None:
            args = parser.parse_args()
            self.__args.update(vars(args))

        else:
            with open(yaml_file, "r") as f:
                args = yaml.safe_load(f)
                self.__args.update(args)

        self.__args.update(kwargs)

        super().__init__(**self.__args)

    def set_configuration(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        self.__init__(parser, yaml_file, **kwargs)

    def reset_configuration(self):
        self.__init__(**vars(self.__args))


class Configuration(_BaseConfiguration):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)


class ModelConfiguration(Configuration):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def create_model(self):
        raise NotImplementedError(
            "`create_model()` method is not implemented. Please implement it in your subclass."
        )


def create_configuration_from_yaml(
    yaml_file: os.PathLike, _configuration_class: _BaseConfiguration = Configuration
):
    """
    Creates a configuration object from a YAML file.

    ## Args:
        `yaml_file` (`os.PathLike`): Path to the YAML file.
        `_configuration_class` (`_BaseConfiguration`): Configuration class to use. Defaults to `Configuration`.

    ## Returns:
        `_BaseConfiguration`: _description_
    """
    with open(yaml_file, "r") as f:
        y = yaml.safe_load(f)

    config: _BaseConfiguration = _configuration_class(**y)
    return config
