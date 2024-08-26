import os
import shutil


class NonExistentConfigFileError(Exception):
    pass


def get_config_file_path(environment_name, algorithm_name):
    file_path = os.path.join('config_files', environment_name, f'{algorithm_name}.txt')

    if not os.path.exists(file_path):
        raise NonExistentConfigFileError(f'Config {file_path} does not exist')

    return file_path


def restore_default_config(environment_name, algorithm_name):
    default_file_path = os.path.join('default_config_files', environment_name, f'{algorithm_name}.txt')
    file_path = os.path.join('config_files', environment_name, f'{algorithm_name}.txt')

    if os.path.exists(file_path):
        os.remove(file_path)

    shutil.copy(default_file_path, file_path)


def restore_all_default_configs():
    DEFAULT_CONFIG_DIRECTORY = 'default_config_files'
    CONFIG_DIRECTORY = 'config_files'

    if os.path.exists(CONFIG_DIRECTORY):
        shutil.rmtree(CONFIG_DIRECTORY)

    shutil.copytree(DEFAULT_CONFIG_DIRECTORY, CONFIG_DIRECTORY)
