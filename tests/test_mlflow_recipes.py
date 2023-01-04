import pytest

# teddytools imports
from teddytools.utils.config import Configuration, create_configuration_from_yaml
from teddytools.utils import repl
from teddytools.sklearn.model import SKLearnModelConfiguration
from teddytools.mlflow.scripts.recipes import (
    create_steps_from_config,
    generate_recipe_template,
)

repl.install()


class TestMLflowRecipes:
    @pytest.fixture
    def run_config(self, run_config_yaml_file):
        run_config = create_configuration_from_yaml(
            yaml_file=run_config_yaml_file, _configuration_class=Configuration
        )
        return run_config

    @pytest.fixture
    def model_config(self, model_config_yaml_file: str):
        model_config = create_configuration_from_yaml(
            yaml_file=model_config_yaml_file,
            _configuration_class=SKLearnModelConfiguration,
        )
        return model_config

    def test_generate_recipe_template(self, run_config, model_config):
        recipe_dict = generate_recipe_template(
            run_config, model_config, recipe_yaml_file="test_artifacts_dir/recipe.yaml"
        )
        assert recipe_dict["steps"] is not None

        # ensure all steps are properly initialized
        steps = recipe_dict["steps"]
        assert steps["ingest"] == "{{INGEST_CONFIG}}"

        # additional register checks
        assert steps["register"] is not None
        assert steps["register"]["allow_non_validated_model"] == False

        assert steps["split"] is not None
        assert steps["transform"] is not None
        assert steps["train"] is not None
        assert steps["evaluate"] is not None
