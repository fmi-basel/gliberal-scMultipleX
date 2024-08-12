import json
from pathlib import Path

from fractal_tasks_core.dev.lib_args_schemas import create_schema_for_single_task
from fractal_tasks_core.dev.lib_signature_constraints import (
    _extract_function,
    _validate_function_signature,
)

import scmultiplex

FRACTAL_TASKS_CORE_DIR = Path(scmultiplex.__file__).parent
PACKAGE_NAME = "scmultiplex"
with (FRACTAL_TASKS_CORE_DIR / "__FRACTAL_MANIFEST__.json").open("r") as f:
    MANIFEST = json.load(f)
TASK_LIST = MANIFEST["task_list"]


def test_task_functions_have_valid_signatures():
    """
    Test that task functions have valid signatures.
    """
    for task in TASK_LIST:
        for key in ["executable_non_parallel", "executable_parallel"]:
            value = task.get(key, None)
            if value is not None:
                function_name = Path(task[key]).with_suffix("").name
                task_function = _extract_function(
                    task[key], function_name, package_name=PACKAGE_NAME
                )
                _validate_function_signature(task_function)


def test_args_schemas_are_up_to_date():
    """
    Test that args_schema attributes in the manifest are up-to-date
    """
    for ind_task, task in enumerate(TASK_LIST):
        for kind in ["non_parallel", "parallel"]:
            key = f"executable_{kind}"
            value = task.get(key, None)
            if value is not None:
                print(f"Now handling {task[key]}")
                old_schema = TASK_LIST[ind_task].get(f"args_schema_{kind}", None)
                assert old_schema is not None
                new_schema = create_schema_for_single_task(
                    task[key], package=PACKAGE_NAME
                )
                # The following step is required because some arguments may
                # have a default which has a non-JSON type (e.g. a tuple),
                # which we need to convert to JSON type (i.e. an array) before
                # comparison.
                new_schema = json.loads(json.dumps(new_schema))
                assert new_schema == old_schema
