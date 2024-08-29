## How to Run The Program
In order to start the program, just run main.py. All other .py files are just modules that are not meant to be run directly. 

## How to Use The Program
The program has a simple GUI with a sidebar that should be easy to use without any specific knowledge. However, here is a quick overview of what all of the GUI elements in the sidebar do:

##### Dropdown Menus
There are two dropdown menus where you can choose the environment you want to train the program to solve and the algorithm that you want to use to solve it. 

##### Buttons Related to Config Files
There are three buttons related to config files: One opens the config file that corresponds to the chosen environment and algorithm in the default text editor, another resets this config file and the last one resets all config files. 

##### Spinbox
There is a spinbox where you can specify the number of generations that you want to train the algorithm for. 

##### Checkbuttons
There are two checkbuttons. One controls whether your computer will use multiprocessing to speed up the training and the other one controls whether statistics will be written into the console as the algorithm is being trained. The format of the statistics vary based on the algorithm you pick. 

##### Main Button
At the bottom of the sidebar, there is the main button. This button will change during the program. At first, it will say "Train" and you can use it to start training the algorithm. Once you click it, it gets deactivated until the algorithm is trained. Once the algorithm is trained, it will change back to active, its text will be "Play Game" and you can press it to watch the algorithm play the game. While the algorithm is playing, the button will be deactivated again and once the game is over, it will change to its initial "Train" state again.

![example image](../images/example-image.png)

### Environments
There are four available training environments from the [Gymnasium](https://gymnasium.farama.org/) library. These are: [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/), [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/), [Mountain Car](https://gymnasium.farama.org/environments/classic_control/mountain_car/) and [Lunar lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/). 

### Algorithms
There are for neuroevolution algorithms available: NEAT (NeuroEvolution of Augmenting Topologies), HyperNEAT (Hypercube-based NEAT), neuroevolution of fixed topologies where networks are optimized by CMA-ES (Covariance Matrix Adaptation Evolution Strategy) and neuroevolution of fixed topologies where networks are optimized by Differential Evolution. 

### Config Files
Every algorithm works with different parameters, and therefore the structure of config files varies. 

#### NEAT config
Config file for the NEAT algorithm works exactly the way it does in neat-python library. An overview can be found [here](https://neat-python.readthedocs.io/en/latest/config_file.html). 

#### HyperNEAT config
HyperNEAT config files work almost the same way as NEAT, except you should no longer specify num_inputs and num_outputs, because now the settings relate to the CPPN network that is evolved by NEAT and the number of its inputs and outputs depend on the dimensionality of its substrate. There is a new section -[Substrate] - where you can specify the number of inputs that the substrate takes, (substrate_num_inputs), the number of outputs that it produces(substrate_num_outputs), number of its hidden layers (substrate_hidden_layers), size of its hidden layers (substrate_layer_size), the activation function used in its hidden layers (activation_substrate_hidden) and the activation function used for its output (activation_substrate_output). 

#### CMA-ES config
CMA-ES config files have a different structure than NEAT and HyperNEAT config files. Here, the settings are not divided into sections and you can specify the number of inputs of the network (num_inputs), the number of outputs of the network (num_outputs), the number of hidden layers (hidden_layers) and their size (hidden_layer_size), the activation function used in the hidden layers (activation_hidden), the activation function used for the output (activation_output), and parameters sigma, mu and lambda (by their names) directly for the CMA-ES algorithm.

#### Differential Evolution config
Differential Evolution config files are similar to CMA-ES, but contain different parameters. The parameters here are: the number of inputs of the network (num_inputs), the number of outputs of the network (num_outputs), the number of hidden layers (hidden_layers) and their size (hidden_layer_size), the activation function used in the hidden layers (activation_hidden), the activation function used for the output (activation_output), population size (population_size) and the bounds between which the algorithm should search for weights (population_size). Note: when you specify a bound, the algorithm initializes the weights in the first population in the range [-bound, bound]. 

### Activation functions
The available activation functions for NEAT networks and HyperNEAT CPPNs are listed [here](https://neat-python.readthedocs.io/en/latest/activation.html). The available activation functions for HyperNEAT substrate and fixed topology networks trained by CMA-ES and Differential evolution are: abs, cube, exp, identity, relu, log, sigmoid, sin, softplus, square and tanh
