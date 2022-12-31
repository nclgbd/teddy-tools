# common imports
from argparse import ArgumentParser

def create_default_parser():
    """
    Creates an argument parser with default arguments.

    ## Returns:
        `ArgumentParser`: Argument parser with default arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--run_config_yaml_path",
        default="tests/config/runconfig/run.yml",
        help="Path to run configuration file.",
        type=str,
    )
    parser.add_argument(
        "--model_config_yaml_path",
        help="Path to model configuration file. If not set here, is inferred from `runconfig` argument.",
        type=str,
    )
    return parser
