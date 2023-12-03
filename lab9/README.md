# Lab 9

The code implements a Genetic Algorithm that tries to maximize value of the provided fitness function, while trying
to reduce the number of calls to that.

## How my solution has been defined

Key concepts and strategies:

- **Population Initialization:** Generates an initial population with random genomes.
  
- **Selection:** Tournament selection to choose parents for mutation and crossover. 
There's a consideration for a "fitness hole" during selection.

- **Crossover and Mutation:** Uniform crossover and single-point mutation operations to create offspring.

- **Simulated Annealing-like Survival Selection:** Implements a survival selection strategy with elements inspired 
by simulated annealing. The strategy involves sorting individuals by fitness and choosing survivors based on either 
classical or simulated annealing-like selection, depending on the improvement in average fitness.

- **Random Restarts:** Introduces a probability of restarting the optimization process with a completely new 
population.

- **Entropic Selection:** Optionally considers individuals with high entropy to diversify the population.
  (default: no entropy selection)

- **Termination Condition:** The algorithm runs for a specified number of epochs or until a fitness value of 1 is 
achieved.
