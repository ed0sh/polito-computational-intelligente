# Lab 2: Nim ES

The code implements an Evolutionary Strategy that tries to extract the best set of rules 
to play the nim game without actually knowing about the _nim sum_ method.

## How my solution has been defined
### Initialization
Create a set of pseudo-random conditions and a set containing all the possible moves that
can be played in a game.<br>
Then, to create the set of rules, pair each condition to each possible move.<br>
When creating the initial population, select randomly `NUM_ROWS` rules for each `NimAgent`, 
the one responsible to play the games.

### Evolution
I defined a finite numer of `EPOCHS` in which the agents can evolve.<br>
At each epoch, a new offspring of `NimAgent` will be created, evaluated against the professor proposed strategies and
a new population will be selected according to each agent fitness (defined as the winning ratio).<br>
Moreover, each rule is associated with a weight, incremented o decremented according to the number of winning games of 
the agent using it. Then, the rule weight is used in crossover and mutation as a weight for the random selection mechanism.

### Termination
At the end of the `EPOCHS`, I select the best ranking rules from the global set of rules 
to create the best rule-ranked agent that plays the Nim game
