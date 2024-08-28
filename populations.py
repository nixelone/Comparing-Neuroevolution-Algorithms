"""
This module contains implementations of evolutionary
algorithms used to train neural networks

For every algorithm, there is an object representing a population
of individuals that can be trained using that algorithm

All population objects inherit from class
NeuralNetworkPopulation for consistency

There is also a dictionary where all population classes
can be accessed using the algorithm's string name
"""

import neat
from abc import ABC
from abc import abstractmethod
import multiprocessing
from functools import partial
import numpy as np
from deap import algorithms
from deap import creator
from deap import base
from deap import cma
from deap import tools
from scipy.optimize import differential_evolution

from networks import NEATNetwork
from networks import HyperNEATNetwork2D
from networks import FixedTopologyNetwork
from config_processor import create_hyperneat_cppn_config
from config_processor import create_hyperneat_substrate_config
from config_processor import create_neft_network_config
from config_processor import create_cmaes_config
from config_processor import create_de_config


class NoWinnerNetworkError(Exception):
    """
    Exception that is raised when there is an attempt
    at getting the most fit network from a population
    that has not been trained yet
    """
    pass


class InvalidCentroidError(Exception):
    """
    Exception that is raised when a centroid of
    invalid length is specified in CMA-ES algorithm
    """
    pass


class NeuralNetworkPopulation(ABC):
    """
    Abstract class that all populations inherit from
    """

    @abstractmethod
    def train(self, fitness_function, n_generations, n_processes):
        pass

    def fit(self, fitness_function, n_generations, n_processes):
        self.train(fitness_function, n_generations, n_processes)

    @abstractmethod
    def get_winning_network(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NEATTypePopulation(NeuralNetworkPopulation, ABC):
    """
    Class that NEATPopulation and HyperNEATPopulation2D
    inherit from that contains all of their shared parts
    """

    def __init__(self, config_file, reporter):
        """
        Initializes variables that belong to the class
        """
        config = self._process_config(config_file)
        self._population = neat.Population(config)
        if reporter:
            stdout_reporter = neat.StdOutReporter(show_species_detail=False)
            self._population.add_reporter(stdout_reporter)

    @staticmethod
    @abstractmethod
    def _process_config(config_file):
        """
        Processes config file to get settings for
        the population and returns config object

        Every child class has to provide its
        own implementation of this method
        """
        pass

    @abstractmethod
    def _create_network(self, genome, config):
        """
        Creates and returns neural network from the genome

        Every child class has to provide its
        own implementation of this method
        """
        pass

    def _evaluate_genome(self, genome, config, fitness_function):
        """
        Evaluates and returns the fitness of given genome
        """
        network = self._create_network(genome, config)
        return fitness_function(network)

    def _create_fitness_function(self, fitness_function, n_processes):
        """
        Creates and returns a function that evaluates the fitness of the whole population
        """

        def population_fitness_function(genomes, config):
            """
            Evaluates and assigns the fitness to every individual in the population

            Uses multiprocessing to speed up the evaluation
            """
            eval_function = partial(self._evaluate_genome,
                                    config=config, fitness_function=fitness_function)
            with multiprocessing.Pool(processes=n_processes) as pool:
                fitnesses = pool.map(eval_function, [genome for _, genome in genomes])

            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

        return population_fitness_function

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):
        """
        Trains the population for specified number of
        generations using provided fitness function
        """
        population_fitness_function = self._create_fitness_function(fitness_function, n_processes)
        self._population.run(population_fitness_function, n_generations)

    def get_winning_network(self):
        """
        Returns the fittest network that has been found

        Raises an exception if the population has not been trained yet
        """
        if self._population.generation > 0:
            winning_genome = self._population.best_genome
            winning_network = self._create_network(winning_genome, self._population.config)
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')


class NEATPopulation(NEATTypePopulation):
    """
    Class that represents a population of neural
    networks that can be trained using NEAT algorithm
    """

    def __init__(self, config_file, reporter=False):
        """
        Initializes variables that belong to the class
        """
        super().__init__(config_file, reporter)

    @staticmethod
    def _process_config(config_file):
        """
        Processes config file to get settings for
        the population and returns config object
        """

        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_file
        )

        return config

    def _create_network(self, genome, config):
        """
        Creates and returns neural network from the genome
        """
        population_size = len(self._population.population)
        generation = self._population.generation

        network = NEATNetwork(genome, config, population_size, generation)
        return network

    def __str__(self):
        """
        Returns a string with information about the population

        Is called when a NEATPopulation object is converted to string
        """
        return f'A NEAT population of {len(self._population.population)} individuals ' \
               f'that has been trained for {self._population.generation} generations'


class HyperNEATPopulation2D(NEATTypePopulation):
    """
    Class that represents a population of neural networks
    that can be trained using HyperNEAT algorithm
    """

    def __init__(self, config_file, reporter=False):
        """
        Initializes variables that belong to the class and
        creates a config object for the substrate network
        """
        super().__init__(config_file, reporter)
        self.substrate_config = create_hyperneat_substrate_config(config_file)

    @staticmethod
    def _process_config(config_file):
        """
        Processes config file to get settings for
        the population and returns config object
        """
        cppn_config = create_hyperneat_cppn_config(config_file, n_substrate_dimensions=2)
        return cppn_config

    def _create_network(self, genome, config):
        """
        Creates and returns a substrate neural network from the genome
        """
        population_size = len(self._population.population)
        generation = self._population.generation

        network = HyperNEATNetwork2D(genome, config, self.substrate_config, population_size, generation)
        return network

    def __str__(self):
        """
        Returns a string with information about the population

        Is called when a HyperNEATPopulation2D object is converted to string
        """
        return f'A HyperNEAT population of 2-dimensional networks ' \
               f'of {len(self._population.population)} individuals ' \
               f'that has been trained for {self._population.generation} generations'


class CMAESNetworkPopulation(NeuralNetworkPopulation):
    """
    Class that represents a population of fixed topology neural
    networks that can be trained using CMA-ES algorithm
    """

    def __init__(self, config_file, centroid=None, reporter=False):
        """
        Initializes variables that belong to the class,
        initializes DEAP Toolbox with parameters that correspond
        to CMA-ES algorithm

        Raises an exception if a centroid of invalid length is specified
        """

        self._network_config = create_neft_network_config(config_file)
        self._cmaes_config = create_cmaes_config(config_file)

        network_size = FixedTopologyNetwork.get_vectorized_size(self._network_config)
        if centroid is None:
            centroid = np.zeros(network_size)
        elif len(centroid) != network_size:
            raise InvalidCentroidError(f'Length of the centroid should be {network_size}')

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()

        # use CMA-ES
        strategy = cma.Strategy(
            centroid=centroid,
            sigma=self._cmaes_config.sigma,
            lambda_=self._cmaes_config.lambda_,
            mu=self._cmaes_config.mu
        )
        self._toolbox.register('generate', strategy.generate, creator.Individual)
        self._toolbox.register('update', strategy.update)

        self._stats = None
        self._verbose = False
        if reporter:
            self._enable_reporting()

        # initialize hall of fame so the best individual can be retrieved
        self._hall_of_fame = tools.HallOfFame(1, similar=np.array_equal)
        self._n_generations = 0

    def _enable_reporting(self):
        """
        Enables reporting of statistics in the console during training
        """

        self._stats = tools.Statistics(lambda individual: individual.fitness.values)
        self._stats.register('avg', np.mean)
        self._stats.register('std', np.std)
        self._stats.register('min', np.min)
        self._stats.register('max', np.max)
        self._verbose = True

    @staticmethod  # this method has to be static in order to be pickled for multiprocessing
    def _evaluate_fitness(weight_vector, fitness_function, network_config):
        """
        Evaluates and returns the fitness of a neural network
        """

        network = FixedTopologyNetwork(weight_vector, network_config, algorithm_name='cma-es')

        # return a tuple, as required by DEAP
        return fitness_function(network),

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):
        """
        Trains the population for specified number of
        generations using provided fitness function

        Uses multiprocessing to speed up the process
        """

        eval_function = partial(self._evaluate_fitness, fitness_function=fitness_function, network_config=self._network_config)
        self._toolbox.register('evaluate', eval_function)

        with multiprocessing.Pool(processes=n_processes) as pool:
            self._toolbox.register('map', pool.map)

            algorithms.eaGenerateUpdate(
                self._toolbox,
                ngen=n_generations,
                halloffame=self._hall_of_fame,
                stats=self._stats,
                verbose=self._verbose)

            self._toolbox.unregister('map')

        self._toolbox.unregister('evaluate')
        self._n_generations += n_generations

    def get_winning_network(self):
        """
        Returns the fittest network that has been found

        Raises an exception if the population has not been trained yet
        """

        if self._n_generations > 0:
            winning_genome = self._hall_of_fame[0]
            winning_network = FixedTopologyNetwork(winning_genome, self._network_config, algorithm_name='cma-es')
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')

    def __str__(self):
        """
        Returns a string with information about the population
        Is called when a CMAESNetworkPopulation object is converted to string
        """
        return f'A NEFT population of {self._cmaes_config.mu} individuals that has been trained ' \
               f'for {self._n_generations} generations by CMA-ES algorithm'


class DifferentialEvolutionNetworkPopulation(NeuralNetworkPopulation):
    """
    Class that represents a population of fixed topology neural
    networks that can be trained using differential evolution
    """

    def __init__(self, config_file, reporter=False):
        """
        Initializes variables that belong to the class
        """

        self._network_config = create_neft_network_config(config_file)
        self._de_config = create_de_config(config_file)

        network_size = FixedTopologyNetwork.get_vectorized_size(self._network_config)
        self._bounds = [(-self._de_config.bound, self._de_config.bound)] * network_size

        self._callback = None
        if reporter:
            self._callback = self._reporting_function

        self._population = None
        self._winning_genome = None
        self._n_generations = 0

    def _evaluate_fitness(self, weight_vector, fitness_function):
        """
        Evaluates and returns the fitness of a neural network
        """

        network = FixedTopologyNetwork(weight_vector, self._network_config, algorithm_name='differential evolution')
        return -fitness_function(network)

    def _reporting_function(self, _, convergence):
        """
        Reports information about the population during training after each generation
        """
        print(f'Generation: {self._n_generations}, Convergence: {convergence}')
        self._n_generations += 1

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):
        """
        Trains the population for specified number of
        generations using provided fitness function

        Uses multiprocessing to speed up the process
        """

        init = 'latinhypercube'
        if self._population is not None:
            init = self._population

        eval_function = partial(self._evaluate_fitness, fitness_function=fitness_function)
        result = differential_evolution(
            eval_function,
            self._bounds,
            strategy='best1bin',
            maxiter=n_generations,
            popsize=self._de_config.population_size,
            init=init,
            workers=n_processes,
            callback=self._callback
        )
        self._n_generations += n_generations

        self._population = result.population
        self._winning_genome = np.array(result.x)

    def get_winning_network(self):
        """
        Returns the fittest network that has been found

        Raises an exception if the population has not been trained yet
        """

        if self._winning_genome is not None:
            winning_network = FixedTopologyNetwork(self._winning_genome, self._network_config, algorithm_name='differential evolution')
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')

    def __str__(self):
        """
        Returns a string with information about the population
        Is called when a DifferentialEvolutionNetworkPopulation object is converted to string
        """

        return f'A NEFT population of {self._de_config.population_size} individuals that has been trained ' \
               f'for {self._n_generations} generations by differential evolution'


populations = {
    'NEAT': NEATPopulation,
    'HyperNEAT': HyperNEATPopulation2D,
    'CMA-ES': CMAESNetworkPopulation,
    'Differential Evolution': DifferentialEvolutionNetworkPopulation
}
