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
    pass


class InvalidCentroidError(Exception):
    pass


class NeuralNetworkPopulation(ABC):

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

    def __init__(self, config_file, reporter):
        config = self._process_config(config_file)
        self._population = neat.Population(config)
        if reporter:
            stdout_reporter = neat.StdOutReporter(show_species_detail=False)
            self._population.add_reporter(stdout_reporter)

    @staticmethod
    @abstractmethod
    def _process_config(config_file):
        pass

    @abstractmethod
    def _create_network(self, genome, config):
        pass

    def _evaluate_genome(self, genome, config, fitness_function):
        network = self._create_network(genome, config)
        return fitness_function(network)

    def _create_fitness_function(self, fitness_function, n_processes):

        def population_fitness_function(genomes, config):
            eval_function = partial(self._evaluate_genome,
                                    config=config, fitness_function=fitness_function)
            with multiprocessing.Pool(processes=n_processes) as pool:
                fitnesses = pool.map(eval_function, [genome for _, genome in genomes])

            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

        return population_fitness_function

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):
        population_fitness_function = self._create_fitness_function(fitness_function, n_processes)
        self._population.run(population_fitness_function, n_generations)

    def get_winning_network(self):
        if self._population.generation > 0:
            winning_genome = self._population.best_genome
            winning_network = self._create_network(winning_genome, self._population.config)
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')


class NEATPopulation(NEATTypePopulation):

    def __init__(self, config_file, reporter=False):
        super().__init__(config_file, reporter)

    @staticmethod
    def _process_config(config_file):
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation,
            config_file
        )

        return config

    def _create_network(self, genome, config):
        population_size = len(self._population.population)
        generation = self._population.generation

        network = NEATNetwork(genome, config, population_size, generation)
        return network

    def __str__(self):
        return f'A NEAT population of {len(self._population.population)} individuals ' \
               f'that has been trained for {self._population.generation} generations'


class HyperNEATPopulation2D(NEATTypePopulation):

    def __init__(self, config_file, reporter=False):
        super().__init__(config_file, reporter)
        self.substrate_config = create_hyperneat_substrate_config(config_file)

    @staticmethod  # should it be a classmethod when I am not using cls? Why not find a way to make it static, even though it is abstract
    def _process_config(config_file):
        cppn_config = create_hyperneat_cppn_config(config_file, n_substrate_dimensions=2)
        return cppn_config

    def _create_network(self, genome, config):
        population_size = len(self._population.population)
        generation = self._population.generation

        network = HyperNEATNetwork2D(genome, config, self.substrate_config, population_size, generation)
        return network

    def __str__(self):
        return f'A HyperNEAT population of 2-dimensional networks ' \
               f'of {len(self._population.population)} individuals ' \
               f'that has been trained for {self._population.generation} generations'


class CMAESNetworkPopulation(NeuralNetworkPopulation):

    def __init__(self, config_file, centroid=None, reporter=False):

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

        self._hall_of_fame = tools.HallOfFame(1, similar=np.array_equal)
        self._n_generations = 0

    def _enable_reporting(self):
        self._stats = tools.Statistics(lambda individual: individual.fitness.values)
        self._stats.register('avg', np.mean)
        self._stats.register('std', np.std)
        self._stats.register('min', np.min)
        self._stats.register('max', np.max)
        self._verbose = True

    @staticmethod  # this method has to be static in order to be pickled for multiprocessing
    def _evaluate_fitness(weight_vector, fitness_function, network_config):
        network = FixedTopologyNetwork(weight_vector, network_config, algorithm_name='cma-es')
        return fitness_function(network),  # the function returns a tuple as required by deap

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):

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
        if self._n_generations > 0:
            winning_genome = self._hall_of_fame[0]
            winning_network = FixedTopologyNetwork(winning_genome, self._network_config, algorithm_name='cma-es')
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')

    def __str__(self):
        return f'A NEFT population of {self._cmaes_config.mu} individuals that has been trained ' \
               f'for {self._n_generations} generations by CMA-ES algorithm'


class DifferentialEvolutionNetworkPopulation(NeuralNetworkPopulation):

    def __init__(self, config_file, reporter=False):

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
        network = FixedTopologyNetwork(weight_vector, self._network_config, algorithm_name='differential evolution')
        return -fitness_function(network)

    def _reporting_function(self, _, convergence):
        print(f'Generation: {self._n_generations}, Convergence: {convergence}')
        self._n_generations += 1

    def train(self, fitness_function, n_generations, n_processes=multiprocessing.cpu_count()):

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
        if self._winning_genome is not None:
            winning_network = FixedTopologyNetwork(self._winning_genome, self._network_config, algorithm_name='differential evolution')
            return winning_network
        else:
            raise NoWinnerNetworkError('The population has not been trained yet')

    def __str__(self):
        return f'A NEFT population of {self._de_config.population_size} individuals that has been trained ' \
               f'for {self._n_generations} generations by differential evolution'


populations = {
    'NEAT': NEATPopulation,
    'HyperNEAT': HyperNEATPopulation2D,
    'CMA-ES': CMAESNetworkPopulation,
    'Differential Evolution': DifferentialEvolutionNetworkPopulation
}
