"""
This module contains functions that extract information from
config files and dataclasses that store this information
"""

import re
import tempfile
import neat
import os
from dataclasses import dataclass


class DuplicateOptionError(Exception):
    """
    Exception that is raised when a parameter is
    listed more than once in a config file
    """
    pass


@dataclass
class FixedTopologyNetworkConfig:
    """
    Dataclass that holds information about
    a fixed topology network
    """
    num_inputs: int
    num_outputs: int
    hidden_layers: int
    hidden_layer_size: int
    activation_hidden: str
    activation_output: str


@dataclass
class CMAESPopulationConfig:
    """
    Dataclass that holds information about
    an instance of CMA-ES algorithm
    """
    mu: int
    lambda_: int
    sigma: float


@dataclass
class DifferentialEvolutionConfig:
    """
    Dataclass that holds information about an
    instance of Differential Evolution algorithm
    """
    population_size: int
    bound: float


def extract_value(variable_name, variable_type, config_file):
    """
    Extracts value corresponding to a given name from a given
    config file and tries to return it as specified type

    Raises an exception when the item is not listed
    or if it is listed more than once

    An exception is also raised is the listed value
    cannot be converted to the specified type
    """
    with open(config_file, 'r') as f:
        config_text = f.read()

    occurences = re.findall(fr'(?:^|\n){variable_name}\s*=\s*([-+]?\d+(?:\.\d*)?|[a-zA-Z]+)', config_text)
    if len(occurences) == 0:
        raise RuntimeError(f'Missing configuration item: {variable_name}')
    elif len(occurences) > 1:
        raise DuplicateOptionError(f'Option \'{variable_name}\' is listed more than once')

    search_result = variable_type(occurences[0])
    return search_result


def check_line_name(names, line):
    """
    Checks for name of the variable that is in the given string,
    returns True if at least one of the specified names
    is the string variable name

    Names can be either a string (one name) or a list of strings (multiple names)
    """
    # handles inputs that are given in string format
    # instead of a list of strings
    if type(names) == str:
        names = [names]

    for name in names:
        searched_name = re.escape(name)
        if re.search(fr'^{searched_name}[= \n]', line):
            return True
    return False


def create_hyperneat_substrate_config(config_file):
    """
    Creates and returns FixedTopologyNetworkConfig for
    HyperNEAT substrate based on the given config file
    """
    substrate_num_inputs = extract_value('substrate_num_inputs', int, config_file)
    substrate_num_outputs = extract_value('substrate_num_outputs', int, config_file)
    substrate_hidden_layers = extract_value('substrate_hidden_layers', int, config_file)
    substrate_layer_size = extract_value('substrate_layer_size', int, config_file)
    activation_substrate_hidden = extract_value('activation_substrate_hidden', str, config_file)
    activation_substrate_output = extract_value('activation_substrate_output', str, config_file)

    config = FixedTopologyNetworkConfig(
        num_inputs=substrate_num_inputs,
        num_outputs=substrate_num_outputs,
        hidden_layers=substrate_hidden_layers,
        hidden_layer_size=substrate_layer_size,
        activation_hidden=activation_substrate_hidden,
        activation_output=activation_substrate_output
    )
    return config


def create_hyperneat_cppn_config(config_file, n_substrate_dimensions):
    """
    Creates and returns config for HyperNEAT CPPN that is evolved
    by NEAT algorithm based on the specified config file
    """
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp_file:
        temp_filename = temp_file.name

        with open(config_file, 'r') as f:

            irrelevant_parameters = (
                'num_inputs',
                'num_outputs'
            )

            for line in f:
                if check_line_name('[DefaultGenome]', line):
                    temp_file.write(line)
                    temp_file.write(f'num_inputs = {n_substrate_dimensions * 2}\n')
                    temp_file.write('num_outputs = 1\n')
                elif not check_line_name(irrelevant_parameters, line):
                    temp_file.write(line)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        temp_filename)

    os.remove(temp_filename)
    return config


def create_neft_network_config(config_file):
    """
    Creates and returns FixedTopologyNetworkConfig for fixed
    topology neural network based on the given config file
    """
    num_inputs = extract_value('num_inputs', int, config_file)
    num_outputs = extract_value('num_outputs', int, config_file)
    hidden_layers = extract_value('hidden_layers', int, config_file)
    hidden_layer_size = extract_value('hidden_layer_size', int, config_file)
    activation_hidden = extract_value('activation_hidden', str, config_file)
    activation_output = extract_value('activation_output', str, config_file)

    config = FixedTopologyNetworkConfig(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        hidden_layers=hidden_layers,
        hidden_layer_size=hidden_layer_size,
        activation_hidden=activation_hidden,
        activation_output=activation_output
    )
    return config


def create_cmaes_config(config_file):
    """
    Creates and returns CMAESPopulationConfig for
    CMA-ES algorithm based on the given config file
    """
    mu = extract_value('mu', int, config_file)
    lambda_ = extract_value('lambda', int, config_file)
    sigma = extract_value('sigma', float, config_file)

    config = CMAESPopulationConfig(
        mu=mu,
        lambda_=lambda_,
        sigma=sigma
    )
    return config


def create_de_config(config_file):
    """
    Creates and returns DifferentialEvolutionConfig for
    differential evolution based on the given config file
    """
    population_size = extract_value('population_size', int, config_file)
    bound = extract_value('bound', float, config_file)

    config = DifferentialEvolutionConfig(
        population_size=population_size,
        bound=bound
    )
    return config
