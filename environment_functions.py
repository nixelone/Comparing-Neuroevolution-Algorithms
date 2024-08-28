import numpy as np
import gymnasium as gym
from populations import populations
from config_manipulator import get_config_file_path


def acrobot_fitness_function(network):

    env = gym.make('Acrobot-v1')
    env.reset()

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    uppermost = 3 - observation[0] - observation[0] * observation[2] + observation[1] * observation[3]
    # constant 3 chosen so that the lowest possible value of uppermost is 0

    time = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, reward, terminated, truncated, info = env.step(action)

        uppermost = max(uppermost, 3 - observation[0] - observation[0] * observation[2] + observation[1] * observation[3])
        time += 0.002  # constant is chosen so that the lowest possible fitness is 0

        if terminated or truncated:
            done = True

    env.close()

    return 1 + float(uppermost) - time
    # constant 1 is chosen so that the lowest possible fitness is 0
    # uppermost must be converted to float because of an inconsistency in gymnasium library


def cart_pole_fitness_function(network):

    env = gym.make('CartPole-v1')
    env.reset()

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    fitness = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, reward, terminated, truncated, info = env.step(action)

        fitness += reward

        if terminated or truncated:
            done = True

    env.close()

    return fitness


def mountain_car_fitness_function(network):  # TODO: fix negative fitness values (they are still there)

    env = gym.make('MountainCar-v0')
    env.reset()

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    rightmost = observation[0]

    time = 0
    done = False
    while not done:
        action = np.argmax(network(observation))
        observation, reward, terminated, truncated, info = env.step(action)

        rightmost = max(rightmost, observation[0])
        time += 0.005  # constant chosen so that the lowest fitness will be 0

        if terminated or truncated:
            done = True

    env.close()

    return 1.6 + float(rightmost) - time


def lunar_lander_fitness_function(network):

    env = gym.make('LunarLander-v2')

    NUM_EVALUATIONS = 10
    fitnesses = np.zeros(NUM_EVALUATIONS)
    for i in range(NUM_EVALUATIONS):
        env.reset()

        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

        done = False
        while not done:
            action = np.argmax(network(observation))
            observation, reward, terminated, truncated, info = env.step(action)

            fitnesses[i] += reward

            if terminated or truncated:
                done = True

    env.close()

    return max(0, 500 + np.average(fitnesses))


fitness_functions = {
    'Acrobot': acrobot_fitness_function,
    'Cart Pole': cart_pole_fitness_function,
    'Mountain Car': mountain_car_fitness_function,
    'Lunar Lander': lunar_lander_fitness_function
}


def evolve_network(environment_name, algorithm_name, n_generations, multiprocessing, reporter):
    population = populations[algorithm_name](get_config_file_path(environment_name, algorithm_name), reporter=reporter)

    if multiprocessing:
        population.train(fitness_functions[environment_name], n_generations=n_generations)
    else:
        population.train(fitness_functions[environment_name], n_generations=n_generations, n_processes=1)

    winner_net = population.get_winning_network()

    return winner_net


environment_names = {
    'Acrobot': 'Acrobot-v1',
    'Cart Pole': 'CartPole-v1',
    'Mountain Car': 'MountainCar-v0',
    'Lunar Lander': 'LunarLander-v2'
}


def render_game(game_name, network):

    env = gym.make(environment_names[game_name], render_mode='rgb_array')
    env.reset()

    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    done = False
    while not done:
        yield env.render()

        action = np.argmax(network(observation))
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            done = True

    env.close()
    yield None
