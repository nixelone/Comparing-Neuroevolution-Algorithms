"""
This module contains implementations of neural networks
needed for the neuroevolution algorithms

The networks are represented as objects, all networks inherit
from NeuralNetwork class for consistency
"""

from neat.nn.feed_forward import FeedForwardNetwork
from abc import ABC
from abc import abstractmethod
import numpy as np

from activation_functions import functions


class NonExistentNodeError(Exception):
    """
    Exception that is raised when _calculate_node_position is called
    on a node position that does not exist in HyperNEAT network
    """
    pass


class InvalidNetworkVectorError(Exception):
    """
    Exception that is raised when a vector that represents
    a network's weights and biases is of incorrect size
    """
    pass


class NeuralNetwork(ABC):
    """
    Abstract class that all networks inherit from
    """

    @abstractmethod
    def __call__(self, input_data):
        pass

    def activate(self, input_data):
        return self(input_data)

    def predict(self, input_data):
        return self(input_data)

    @abstractmethod
    def __str__(self):
        pass


class NEATNetwork(NeuralNetwork):
    """
    Class that inherits from NeuralNetwork class
    and wraps neat-python network in order to make it
    consistent with other types of networks
    """

    def __init__(self, genome, config, training_individuals, training_generations):
        """
        Initializes variables that belong to the class
        """
        self._network = FeedForwardNetwork.create(genome, config)
        self._training_individuals = training_individuals
        self._training_generations = training_generations

    def __call__(self, input_data):
        """
        Returns output of the network that
        corresponds to the given input
        """
        output_data = self._network.activate(input_data)
        return output_data

    def __str__(self):
        """
        Returns a string with information about the network

        This method is called when NEATNetwork object is converted to string
        """
        return f'A network created by the NEAT algoritm that is a result ' \
               f'of training {self._training_individuals} individuals ' \
               f'for {self._training_generations} generations'


class HyperNEATNetwork2D(NeuralNetwork):
    """
    Class that represents a HyperNEAT network and
    inherits from NeuralNetwork class in order
    to be consistent with other types of networks
    """

    def __init__(self, genome, cppn_config, substrate_config, training_individuals, training_generations):
        """
        Initializes variables that belong to the class
        and finds weights of the network
        """
        self._cppn = FeedForwardNetwork.create(genome, cppn_config)
        self._substrate_config = substrate_config
        self._training_individuals = training_individuals
        self._training_generations = training_generations
        self._input_weights = self._find_input_weights()
        self._hidden_weights = self._find_hidden_weights()
        self._output_weights = self._find_output_weights()
        self._hidden_activation_function = functions[self._substrate_config.activation_hidden]
        self._output_activation_function = functions[self._substrate_config.activation_output]

    def _calculate_node_position(self, node_layer, node_number):
        """
        Takes the position of the node in the network and calculates
        and returns its geometric position normalized between -1 and 1
        """

        # inside a layer, nodes are numbered from top of the layer
        # layer 0 is the input, (n+1)-th layer is the output layer

        max_layer_number = self._substrate_config.hidden_layers + 1

        # if the node is in one of the hidden layers
        if 0 < node_layer < max_layer_number:
            max_node_number = self._substrate_config.hidden_layer_size - 1

        # if the node is in the input layer
        elif node_layer == 0:
            max_node_number = self._substrate_config.num_inputs - 1

        # if the node is in the output layer
        elif node_layer == max_layer_number:
            max_node_number = self._substrate_config.num_outputs - 1

        # if the node is not in an existing layer
        else:
            raise NonExistentNodeError(f'Node layer {node_layer} does not exist in the network')

        # normalize the node coordinates between -1 and 1
        x_coordinate = (node_layer / max_layer_number) * 2 - 1

        if node_number > max_node_number:
            raise NonExistentNodeError(f'Node {node_number} does not exist in layer {node_layer}')
        elif max_node_number > 0:
            y_coordinate = (node_number / max_node_number) * 2 - 1
        else:
            # if there is only one node in the layer, set its y-coordinate to the middle
            y_coordinate = 0

        return x_coordinate, y_coordinate

    def _find_input_weights(self):
        """
        Prompts the CPPN network to get the input weights
        of the substrate network and returns them
        """

        input_layer_size = self._substrate_config.num_inputs
        hidden_layer_size = self._substrate_config.hidden_layer_size

        input_weights = np.zeros((hidden_layer_size, input_layer_size))
        for weight_input in range(input_layer_size):
            for weight_output in range(hidden_layer_size):
                in_x, in_y = self._calculate_node_position(node_layer=0, node_number=weight_input)
                out_x, out_y = self._calculate_node_position(node_layer=1, node_number=weight_output)
                cppn_input = [in_x, in_y, out_x, out_y]

                cppn_output = self._cppn.activate(cppn_input)[0]
                input_weights[weight_output, weight_input] = cppn_output

        return input_weights

    def _find_hidden_weights(self):
        """
        Prompts the CPPN network to get the weights
        between hidden layers of the substrate
        network and returns them
        """
        n_hidden_layers = self._substrate_config.hidden_layers
        hidden_layer_size = self._substrate_config.hidden_layer_size

        hidden_weights = np.zeros((n_hidden_layers - 1, hidden_layer_size, hidden_layer_size))
        for layer_number in range(1, n_hidden_layers):
            for weight_input in range(hidden_layer_size):
                for weight_output in range(hidden_layer_size):
                    in_x, in_y = self._calculate_node_position(layer_number, weight_input)
                    out_x, out_y = self._calculate_node_position(layer_number + 1, weight_output)
                    cppn_input = [in_x, in_y, out_x, out_y]

                    cppn_output = self._cppn.activate(cppn_input)[0]
                    hidden_weights[layer_number - 1, weight_output, weight_input] = cppn_output

        return hidden_weights

    def _find_output_weights(self):
        """
        Prompts the CPPN network to get the output weights
        of the substrate network and returns them
        """
        hidden_layer_size = self._substrate_config.hidden_layer_size
        output_layer_size = self._substrate_config.num_outputs
        output_layer_number = self._substrate_config.hidden_layers + 1

        input_weights = np.zeros((output_layer_size, hidden_layer_size))
        for weight_input in range(hidden_layer_size):
            for weight_output in range(output_layer_size):
                in_x, in_y = self._calculate_node_position(node_layer=output_layer_number - 1, node_number=weight_input)
                out_x, out_y = self._calculate_node_position(node_layer=output_layer_number, node_number=weight_output)
                cppn_input = [in_x, in_y, out_x, out_y]

                cppn_output = self._cppn.activate(cppn_input)[0]
                input_weights[weight_output, weight_input] = cppn_output

        return input_weights

    def __call__(self, input_data):
        """
        Does forward propagation to calculate
        the output of the network corresponding
        to the given input and returns it
        """

        if len(input_data) != self._substrate_config.num_inputs:
            raise RuntimeError(f'Expected {self._substrate_config.num_inputs} inputs, got {len(input_data)}')

        x = np.dot(self._input_weights, input_data)
        x = self._hidden_activation_function(x)

        for i in range(len(self._hidden_weights)):
            x = np.dot(self._hidden_weights[i], x)
            x = self._hidden_activation_function(x)

        x = np.dot(self._output_weights, x)
        x = self._output_activation_function(x)

        return x

    def get_cppn(self):
        """
        Returns the CPPN that produced weights for the substrate
        """
        return self._cppn

    def __str__(self):
        """
        Returns a string with information about the network

        This method is called when a HyperNEATNetwork2D object is converted to string
        """
        hidden_layers = self._substrate_config.hidden_layers
        layer_nodes = self._substrate_config.hidden_layer_size

        return f'A 2-dimensional network created by the HyperNEAT algoritm ' \
               f'that is a result of training {self._training_individuals} ' \
               f'individuals for {self._training_generations} generations ' \
               f'that contains {hidden_layers} hidden layers, each consisting of ' \
               f'{layer_nodes} hidden nodes'


class FixedTopologyNetwork(NeuralNetwork):
    """
    Class that represents a fixed topology neutral
    network and inherits from NeuralNetwork class in order
    to be consistent with other types of networks
    """

    def __init__(self, weight_vector, config, algorithm_name='unknown'):
        """
        Initializes variables that belong to the class
        and finds weights and biases of the network

        Raises an exception if the weight vector is
        of incorrect length
        """

        self._config = config

        expected_vector_size = FixedTopologyNetwork.get_vectorized_size(config)
        if len(weight_vector) != expected_vector_size:
            raise InvalidNetworkVectorError(f'Weight vector should be of length {expected_vector_size}')

        self._input_weights = self._find_input_weights(weight_vector)
        self._hidden_weights = self._find_hidden_weights(weight_vector)
        self._output_weights = self._find_output_weights(weight_vector)
        self._hidden_biases = self._find_hidden_biases(weight_vector)
        self._output_biases = self._find_output_biases(weight_vector)
        self._hidden_activation_function = functions[self._config.activation_hidden]
        self._output_activation_function = functions[self._config.activation_output]
        self._algorithm_name = algorithm_name
        self._vectorized_network = weight_vector

    def _find_input_weights(self, weight_vector):
        """
        Extracts input weights from the
        weight vector and returns them
        """
        input_layer_size = self._config.num_inputs
        hidden_layer_size = self._config.hidden_layer_size

        num_input_weights = input_layer_size * hidden_layer_size

        input_weight_vector = weight_vector[:num_input_weights]

        input_weights = input_weight_vector.reshape((hidden_layer_size, input_layer_size))
        return input_weights

    def _find_hidden_weights(self, weight_vector):
        """
        Extracts weights for connections
        between hidden layers of the network
        from the weight vector and returns them
        """
        hidden_layer_size = self._config.hidden_layer_size
        input_layer_size = self._config.num_inputs

        n_hidden_layers = self._config.hidden_layers

        num_input_weights = input_layer_size * hidden_layer_size
        num_hidden_weights = (n_hidden_layers - 1) * hidden_layer_size * hidden_layer_size

        hidden_weight_vector = weight_vector[num_input_weights: num_input_weights + num_hidden_weights]

        hidden_weights = hidden_weight_vector.reshape((n_hidden_layers - 1, hidden_layer_size, hidden_layer_size))
        return hidden_weights

    def _find_output_weights(self, weight_vector):
        """
        Extracts output weights from the
        weight vector and returns them
        """
        hidden_layer_size = self._config.hidden_layer_size
        output_layer_size = self._config.num_outputs

        input_layer_size = self._config.num_inputs
        n_hidden_layers = self._config.hidden_layers

        num_input_weights = input_layer_size * hidden_layer_size
        num_hidden_weights = (n_hidden_layers - 1) * hidden_layer_size * hidden_layer_size
        num_output_weights = hidden_layer_size * output_layer_size

        output_weight_vector = weight_vector[num_input_weights + num_hidden_weights:
                                             num_input_weights + num_hidden_weights + num_output_weights]

        input_weights = output_weight_vector.reshape((output_layer_size, hidden_layer_size))
        return input_weights

    def _find_hidden_biases(self, weight_vector):
        """
        Extracts biases for nodes in hidden layers
        from the weight vector and returns them
        """
        hidden_layer_size = self._config.hidden_layer_size
        n_hidden_layers = self._config.hidden_layers

        output_layer_size = self._config.num_outputs

        num_hidden_biases = n_hidden_layers * hidden_layer_size
        hidden_bias_vector = weight_vector[-num_hidden_biases - output_layer_size: -output_layer_size]

        hidden_biases = hidden_bias_vector.reshape((n_hidden_layers, hidden_layer_size))
        return hidden_biases

    def _find_output_biases(self, weight_vector):
        """
        Extracts biases for nodes in the output layer
        from the weight vector and returns them
        """
        output_layer_size = self._config.num_outputs

        output_biases = weight_vector[-output_layer_size:]
        return output_biases

    @staticmethod
    def get_vectorized_size(config):
        """
        Calculates the size of the vector that represents
        the network's weights and biases
        """
        input_layer_size = config.num_inputs
        hidden_layer_size = config.hidden_layer_size
        n_hidden_layers = config.hidden_layers
        output_layer_size = config.num_outputs

        num_input_weights = input_layer_size * hidden_layer_size
        num_hidden_weights = (n_hidden_layers - 1) * hidden_layer_size * hidden_layer_size
        num_output_weights = hidden_layer_size * output_layer_size

        num_hidden_biases = n_hidden_layers * hidden_layer_size
        num_output_biases = output_layer_size

        return num_input_weights + num_hidden_weights + num_output_weights + num_hidden_biases + num_output_biases

    def __call__(self, input_data):
        """
        Does forward propagation to calculate
        the output of the network corresponding
        to the given input and returns it
        """

        if len(input_data) != self._config.num_inputs:
            raise RuntimeError(f'Expected {self._config.num_inputs} inputs, got {len(input_data)}')

        x = np.dot(self._input_weights, input_data)
        x = x + self._hidden_biases[0]
        x = self._hidden_activation_function(x)

        for i in range(len(self._hidden_weights)):
            x = np.dot(self._hidden_weights[i], x)
            x = x + self._hidden_biases[i + 1]
            x = self._hidden_activation_function(x)

        x = np.dot(self._output_weights, x)
        x = x + self._output_biases
        x = self._output_activation_function(x)

        return x

    def get_vectorized_network(self):
        """
        Returns the vector that represents
        the network's weights and biases
        """
        return self._vectorized_network

    def __str__(self):
        hidden_layers = self._config.hidden_layers
        layer_nodes = self._config.hidden_layer_size

        return f'A fixed-topology network created by {self._algorithm_name} algorithm ' \
               f'that contains {hidden_layers} hidden layers, each consisting of ' \
               f'{layer_nodes} hidden nodes'
