import os
import sys


def extend_at(original: list, index: int, values: list):
    """Extend a list at a specific index with a list of values."""
    return original[:index] + values + original[index:]


def set_sys_arg_defaults_for_hydra(default_config_path: str, project_name_start_index: int = 1, verbose=True, training=True):
    if "++sys_argvs_configured=True" in sys.argv:
        return
    # Extract the mode from the sys.argv (the mode is the positional argument after the .py file)
    py_file_index = min([i for i, arg in enumerate(sys.argv) if arg.endswith(".py")])
    mode_index = py_file_index + 1
    mode = sys.argv.pop(mode_index)
    assert "-" not in mode and "+" not in mode, f"Mode {mode} is not valid. Please specify the mode before the hydra arguments."
    sys.argv.extend([f"++mode={mode}"])

    # Check if the user has specified a config file
    if "-f" in sys.argv:
        idx = sys.argv.index("-f")
        sys.argv[idx] = "--config-name"

    if not "--config-path" in sys.argv:
        # Set the config path and config name, making the config name the last part of the config file path
        if not "--config-name" in sys.argv:
            sys.argv = extend_at(sys.argv, py_file_index + 1, ["--config-name", default_config_path])
        idx = sys.argv.index("--config-name")
        config_path, config_name = sys.argv[idx + 1].rsplit("/", 1)
        sys.argv[idx + 1] = config_name
        sys.argv = extend_at(sys.argv, py_file_index + 1, ["--config-path", config_path])

    config_path = sys.argv[sys.argv.index("--config-path") + 1]
    config_name = sys.argv[sys.argv.index("--config-name") + 1].removesuffix(".yaml").removesuffix(".yml")

    # Check if the user has specified a save_dir
    if "--save-dir" not in sys.argv:
        # AUTOINDEXER_SAVE_DIR overrides the default `./logs` root when checkpoints should land on a larger mounted volume
        sys.argv.extend(["--save-dir", os.environ.get("AUTOINDEXER_SAVE_DIR", "./logs")])

    save_dir_index = sys.argv.index("--save-dir")
    save_dir = os.path.join(sys.argv[save_dir_index + 1], mode)
    sys.argv = sys.argv[:save_dir_index] + sys.argv[save_dir_index + 2:]

    sys.argv.extend([f"++save_dir={save_dir}"])
    if not any("hydra.run.dir" in arg for arg in sys.argv):
        sys.argv.extend([f"hydra.run.dir={save_dir}"])

    # Check if the user has specified a checkpoint
    if "-c" in sys.argv:
        index = sys.argv.index("-c")
        sys.argv[index] = "--checkpoint"

    if "--checkpoint" in sys.argv:
        checkpoint_index = sys.argv.index("--checkpoint")
        checkpoint = sys.argv[checkpoint_index + 1]
        sys.argv = sys.argv[:checkpoint_index] + sys.argv[checkpoint_index + 2:]
        sys.argv.extend([f"++checkpoint={checkpoint}"])

    if training:
        if not any("project_name" in arg for arg in sys.argv):
            # Set project_name to the path of the config file + mode
            if config_path.startswith("./"):
                config_path = config_path[2:]
            project_name = config_path.split("/", project_name_start_index)[-1].replace("/", "-")
            project_name = project_name + "-" + mode
            print(f"Setting project_name to {project_name}")
            sys.argv.extend([f"++project_name={project_name}"])
        if not any("experiment_name" in arg for arg in sys.argv):
            # Set experiment_name to the name of the config file
            print(f"Setting experiment_name to {config_name}")
            sys.argv.extend([f"++experiment_name={config_name}"])

    sys.argv.append("++sys_argvs_configured=True")

    if verbose:
        print("=== sys.argv ===")
        print(sys.argv)
        print("================")
