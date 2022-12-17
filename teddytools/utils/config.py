from argparse import Namespace, ArgumentParser
import yaml
import os


def yaml_to_namespace(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return Namespace(**config)


class _BaseConfiguration(Namespace):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        if parser is not None:
            self.args = parser.parse_args()
        elif yaml_file:
            self.args = yaml_to_namespace(yaml_file)
        elif kwargs:
            self.args = Namespace(**kwargs)

        super().__init__(**vars(self.args))

    def set_configuration(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        self.__init__(parser, yaml_file, **kwargs)

    def reset_configuration(self):
        self.__init__(**vars(self.args))


class Configuration(_BaseConfiguration):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        super().__init__(parser, yaml_file, **kwargs)


class ModelConfiguration(Configuration):
    def __init__(
        self,
        parser: ArgumentParser = None,
        yaml_file: os.PathLike = "",
        **kwargs,
    ):
        super().__init__(parser, yaml_file, **kwargs)

    def create_model(self):
        raise NotImplementedError(
            "`create_model()` method is not implemented. Please implement it in your subclass."
        )
