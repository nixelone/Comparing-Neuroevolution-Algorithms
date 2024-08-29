"""
This module contains functions that can be used for manipulation with config files
"""

import os
import shutil


class NonExistentConfigFileError(Exception):
    """
    Exception that is raised when there is attempt at finding
    a config file that does not exist in the file system
    """
    pass


def get_config_file_path(environment_name, algorithm_name):
    """
    Assembles and returns config file path corresponding to given environment and algorithm

    Raises an exception if the file does not exist
    """
    file_path = os.path.join(
        'config_files',
        environment_name,
        f'{algorithm_name}.txt'
    )

    if not os.path.exists(file_path):
        raise NonExistentConfigFileError(f'Config {file_path} does not exist')

    return file_path


def restore_default_config(environment_name, algorithm_name):
    """
    Restores config file corresponding to given environment and algorithm

    Default config file is taken from default_config_files directory
    """

    default_file_path = os.path.join(
        'default_config_files',
        environment_name,
        f'{algorithm_name}.txt'
    )
    file_path = os.path.join(
        'config_files',
        environment_name,
        f'{algorithm_name}.txt'
    )

    if os.path.exists(file_path):
        os.remove(file_path)

    shutil.copy(default_file_path, file_path)


def restore_all_default_configs():
    """
    Restores all config files

    Default config files are taken from default_config_files directory
    """

    DEFAULT_CONFIG_DIRECTORY = 'default_config_files'
    CONFIG_DIRECTORY = 'config_files'

    if os.path.exists(CONFIG_DIRECTORY):
        shutil.rmtree(CONFIG_DIRECTORY)

    shutil.copytree(DEFAULT_CONFIG_DIRECTORY, CONFIG_DIRECTORY)
