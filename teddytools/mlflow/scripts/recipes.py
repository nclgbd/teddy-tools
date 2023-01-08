import os
import yaml
from teddytools.utils import repl, get_console, get_logger, load_yaml
from teddytools.utils.argparsers import create_default_parser
from teddytools.utils.config import (
    Configuration,
    ModelConfiguration,
    create_configuration_from_yaml,
)
from teddytools.sklearn.model import SKLearnModelConfiguration

_RECIPE_DIRS = ["notebooks", "profiles", "steps"]

console = get_console()
repl.install()
_logger = get_logger("mlflow.recipes")


def _create_recipe_dirs():

    cwd = os.getcwd()
    recipe_paths = []
    for recipe_dir in _RECIPE_DIRS:
        recipe_path = os.path.join(cwd, recipe_dir)
        recipe_paths.append(recipe_path)
        os.makedirs(recipe_path, exist_ok=True)

    return recipe_paths


def _create_split_step(run_config: Configuration, model_config: ModelConfiguration):
    split_step = {}
    split_step["split_ratios"] = [0.8, 0.1, 0.1]
    return split_step


def _create_transform_step(run_config: Configuration, model_config: ModelConfiguration):
    transform_step = {}
    return transform_step


def _create_train_step(run_config: Configuration, model_config: ModelConfiguration):
    train_step = {}
    return train_step


def _create_evaluate_step(run_config: Configuration, model_config: ModelConfiguration):
    evaluate_step = {}
    return evaluate_step


def _create_register_step(run_config: Configuration, model_config: ModelConfiguration):
    register_step = {"allow_non_validated_model": run_config.allow_non_validated_model}
    return register_step


def create_steps_from_config(
    recipe_dict: dict, run_config: Configuration, model_config: ModelConfiguration
):
    console.log("Creating steps from run and model configuration files...")
    steps = {}

    steps["ingest"] = "{}".format("{{INGEST_CONFIG}}")
    steps["split"] = _create_split_step(run_config, model_config)
    steps["transform"] = _create_transform_step(run_config, model_config)
    steps["train"] = _create_train_step(run_config, model_config)
    steps["evaluate"] = _create_evaluate_step(run_config, model_config)
    steps["register"] = _create_register_step(run_config, model_config)

    recipe_dict["steps"] = steps
    console.log("Steps created from configuration files complete.")
    _logger.debug('recipe_dict["steps"]:\n{}', steps)


def generate_recipe_template(
    run_config: Configuration,
    model_config: ModelConfiguration = None,
    recipe_yaml_file: str = "recipe.yaml",
):
    console.log("Generating recipe template...")
    recipe_dict = run_config.mlflow_settings["recipe-yaml"]

    # # create steps
    # create_steps_from_config(recipe_dict, run_config, model_config)

    # save recipe to "recipe.yaml"
    with open(recipe_yaml_file, "w") as f:
        yaml.dump(recipe_dict, f)

    console.log("Recipe template generated.")
    console.print(recipe_dict)
    return recipe_dict


def generate_recipe_template_parser():
    parser = create_default_parser()
    parser.add_argument(
        "-R",
        "--recipe_yaml_file",
        type=str,
        help="Recipe YAML file save location.",
        default="recipe.yaml",
    )
    args = parser.parse_args()
    run_config = create_configuration_from_yaml(
        args.run_config_yaml_file, Configuration
    )
    model_config_yaml_file = (
        args.model_config_yaml_file
        if "model_config_yaml_file" in vars(args)
        else run_config.model_config_yaml_file
    )
    model_config = create_configuration_from_yaml(
        model_config_yaml_file, SKLearnModelConfiguration
    )
    return {
        "run_config": run_config,
        "model_config": model_config,
        "recipe_yaml_file": args.recipe_yaml_file,
    }


def main():
    script_args = generate_recipe_template_parser()
    generate_recipe_template(**script_args)


if __name__ == "__main__":
    main()
