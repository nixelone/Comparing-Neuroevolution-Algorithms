"""
This module contains functions to interact with the training environment
It also contains a custom fitness function for each environment and a dictionary
where all fitness functions can be accessed by the environment's string name
"""

import numpy as np
import gymnasium as gym

from populations import populations
from config_manipulator import get_config_file_path


def acrobot_fitness_function(network):
    """
    Fitness function that evaluates a network's performance in Acrobot

    The final fitness is the highest point that was reached
    minus a constant times the number of time steps that it took

    The function returns fitness as a float
    """

    env = gym.make('Acrobot-v1')
    env.reset()

    observation, _, terminated, truncated, _ = env.step(env.action_space.sample())

    # constant 3 chosen so that the lowest possible value of uppermost is 0
    # therefore uppermost is scaled to the range [0, 6]
    uppermost = \
        3 - observation[0] - observation[0] * observation[2] + observation[1] * observation[3]

    time = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        # constant 3 is chosen so that 'uppermost' is always non-negative
        uppermost = max(
            uppermost,
            3 - observation[0] - observation[0] * observation[2] + observation[1] * observation[3]
        )
        # constant 0.002 is chosen so that the lowest possible fitness is 0
        time += 0.002

        if terminated or truncated:
            done = True

    env.close()

    # constant 1 is chosen so that the lowest possible fitness is 0
    return 1 + float(uppermost) - time


def cart_pole_fitness_function(network):
    """
    Fitness function that evaluates a network's performance in Cart Pole

    The final fitness is the number of time steps that the pole was kept balanced

    The function returns fitness as a float
    """

    env = gym.make('CartPole-v1')
    env.reset()

    observation, _, terminated, truncated, _ = env.step(env.action_space.sample())

    fitness = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        fitness += 1

        if terminated or truncated:
            done = True

    env.close()

    return fitness


def mountain_car_fitness_function(network):
    """
    Fitness function that evaluates a network's performance in Mountain Car

    The final fitness is the furthest point that was reached (to the right)
    minus a constant times the number of time steps that it took

    The function returns fitness as a float
    """

    env = gym.make('MountainCar-v0')
    env.reset()

    observation, _, terminated, truncated, _ = env.step(env.action_space.sample())

    rightmost = observation[0]

    time = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        rightmost = max(
            rightmost,
            observation[0]
        )
        # constant 0.005 is chosen so that the lowest possible fitness is 0
        time += 0.005

        if terminated or truncated:
            done = True

    env.close()

    # constant 1.6 is chosen so that the lowest possible fitness is 0
    return 1.6 + float(rightmost) - time


def lunar_lander_fitness_function(network):
    """
    Fitness function that evaluates a network's performance in Lunar Lander

    The final fitness is the cumulative reward that was obtained from
    the environment processed so that the value is always non-negative,
    averaged over a certain number of evaluations

    The function returns fitness as a float
    """

    env = gym.make('LunarLander-v2')

    # number of times that the network will be evaluated
    # more evaluations will result in more precise, but slower evaluation
    NUM_EVALUATIONS = 10
    fitnesses = np.zeros(NUM_EVALUATIONS)
    for i in range(NUM_EVALUATIONS):
        env.reset()

        observation, reward, terminated, truncated, _ = env.step(env.action_space.sample())

        done = False
        while not done:
            action = np.argmax(network(observation))
            observation, reward, terminated, truncated, _ = env.step(action)

            fitnesses[i] += reward

            if terminated or truncated:
                done = True

    env.close()

    # max(0, x) is applied so that the fitness is always non-negative
    # constant 500 was chosen experimentally
    return max(0, 500 + np.average(fitnesses))


fitness_functions = {
    'Acrobot': acrobot_fitness_function,
    'Cart Pole': cart_pole_fitness_function,
    'Mountain Car': mountain_car_fitness_function,
    'Lunar Lander': lunar_lander_fitness_function
}


def evolve_network(environment_name, algorithm_name, n_generations, multiprocessing, reporter):
    """
    Creates and evolves a population of networks based on the input parameters

    Returns the most fit network at the end of the last iteration of the evolution
    """
    population = populations[algorithm_name](
        get_config_file_path(environment_name, algorithm_name),
        reporter=reporter
    )

    if multiprocessing:
        population.train(
            fitness_functions[environment_name],
            n_generations=n_generations
        )
    else:
        population.train(
            fitness_functions[environment_name],
            n_generations=n_generations,
            n_processes=1
        )

    winner_net = population.get_winning_network()

    return winner_net


environment_names = {
    'Acrobot': 'Acrobot-v1',
    'Cart Pole': 'CartPole-v1',
    'Mountain Car': 'MountainCar-v0',
    'Lunar Lander': 'LunarLander-v2'
}


def render_game(game_name, network):
    """
    Generates and yields frames from specified game
    that is played by network that is passed as input

    Yields 3d array that represents RGB image of
    the game while the game is running

    Yields None once when the game stops
    """

    env = gym.make(environment_names[game_name], render_mode='rgb_array')
    env.reset()

    observation, _, terminated, truncated, _ = env.step(env.action_space.sample())

    done = False
    while not done:
        yield env.render()

        action = np.argmax(network(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            done = True

    env.close()
    yield None
