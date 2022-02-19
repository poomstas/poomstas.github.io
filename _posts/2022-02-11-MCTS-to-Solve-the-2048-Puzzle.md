---
layout: post
title:  "MCTS to Solve the 2048 Puzzle"
date:   2022-02-11
excerpt: "Use the Monte Carlo Tree Search (MCTS) algorithm to solve the 2048 puzzle"
image: "/images/MCTS_2048_thumbnail.png"


---

## Objective / Problem Statement

Create an agent that solves the 2048 puzzle. In this work, I employ the Monte-Carlo Tree Search (MCTS) algorithm to achieve this objective.

The meta-objective is for me to learn how the algorithm works, and practice implementing it on a fun problem :)

-----------------

## The 2048 Game

2048 is a simple board game. As shown above, the board has 4x4 squares, some of which are occupied by number blocks. The player can press one of the four (up, down, left, right) arrows to shift the blocks in the specified direction. Adjacent blocks with the same number will add together, combining into one block with a value of the two blocks' values added together. If the adjacent blocks do not have the same number, then they will not combine.

The game is considered solved if the largest number on the grid reaches 2048, hence the name. The game terminates unsuccessfully if, after an action is taken and a new block is added, there are no empty blocks left and there are no valid moves available.

Play and get a sense of it [here](https://play2048.co/).

------------------

## The Monte Carlo Tree Search (MCTS) Algorithm

Monte Carlo Tree Search (MCTS) is a probabilistic tree search algorithm that uses repeated random sampling to estimate the value of actions as a game progresses. At every iterative state, stochastic sampling is used to update the estimates of the values of actions available at a given state, and the estimates are then used to expand the tree. Then, based on the expanded game tree, the next action is decided. As the algorithm continues to implement actions that are expected to be optimal, the process of traversing and expanding the tree is repeated at every iteration. MCTS is a probabilistic method because at the core of the algorithm, there is the stochastic sampling that attempts to capture the response of actions without exhaustively examining them. Although it does not guarantee that it will find the optimal series of actions, many applications of MCTS in different fields have demonstrated that it is able to provide a close approximation of the optimal policy.

As the name suggests, MCTS involves a tree-like structure; it is used to represent different states and available actions. An example of a tree structure of the 2048 game is provided in the figure below. In the diagram, the nodes represent different states that the game board could take (nodes are more generally represented as a simple circle than a board). At the top of the tree, we start with the root node, with which represent the state of the board when the game begins. Given this state, the player has four different actions he can take (up, down, left, right), which are visualized in the figure. In the diagram, the available actions are represented using edges, which are solid lines that connect the states. The states of the board resulting from these actions are represented using the child nodes. Once the shifting and merging takes place, a new block is randomly placed on the board, some states of which are shown with dotted line arrows. The dotted lines represent stochastic sampling using the Monte Carlo method. Once MCTS selects/executes an action and a stochastic observation is made, then the corresponding child node becomes the new root node, and its sibling nodes would be discarded.

<center><img src="https://github.com/poomstas/2048_MCTS/raw/main/readme_img/A_2048_MCTS.png" alt="" style="max-width:80%;" /></center>


Each state carries with it an estimate of value that is calculated using Monte Carlo simulations. These values are then used to compare the quality of one action over another. Overall, MCTS consists of four stages: selection, expansion, simulation, and backpropagation. The four processes are repeated at every iteration, one of which is shown as an example in the figure below. The two numbers in the nodes of the example tree represent the number of simulated wins and the number of simulations (“visits”) that particular state has observed. The following sections will provide an overview of the mechanics of each stage using the example tree provided.

<center><img src="https://github.com/poomstas/2048_MCTS/raw/main/readme_img/B_MCTS.png" alt="" style="max-width:90%;" /></center>

### Selection

Selection process points the algorithm to which action is likely to be worth exploring. It takes the current state of the tree and selects decisions down that tree to a future state at a fixed depth. The relative value of different nodes are determined using the UCB equation (explained below), which systematically incorporates both the observed average returns and the uncertainty associated with the estimated average.

```python
        # Select
        while node.untried_moves==[] and node.child_nodes!=[]: # Node is fully expanded and non-terminal
            node = node.select_child_UCT(C=exploration_const)
            state.do_move(node.move)
```

#### Upper Confidence Bound (UCB)

The upper confidence bound (UCB) is used in the MCTS’s selection stage to traverse the tree. UCB is used to balance the selection process between exploration and exploitation. Exploration and exploitation refers to the challenge posed to an agent to choose between acquiring new knowledge about the system and returning to an option that is expected to have large returns, based on current knowledge. The UCB algorithm proposes that the agent pull the arm that maximizes the following:

<center><img src="https://github.com/poomstas/2048_MCTS/raw/main/readme_img/C_UcbEqn.png" alt="" style="max-width:80%;" /></center>

The above equation is intuitive. The `ωi/ni` term is the current estimate of returns associated with a decision. The remaining `c sqrt(ln(Ni)/ni)` term represent the upper bound of the confidence interval associated with the estimate, which is updated as the number of observations accumulate over time. The second term decreases as more observations are sampled to represent increased confidence in the expected returns.

The parameter `c` in the above equation controls for the extent to which uncertain options are favored. If `c` is set to zero, then the UCB algorithm would recommend that the agent pulls the arm solely based on the expected returns without considering the uncertainties associated with the estimations (i.e. exploitation). If `c` is set to a larger value, then the relative contribution of the expected value is reduced in the selection process, shifting the significance more towards the uncertainties associated with estimates of expected returns. In other words, a UCB algorithm with a larger `c` tends to favor options that are not previously explored (i.e. exploration). An implementation of the selection process using the UCB equation will show that the algorithm will attempt to quickly identify the best alternative, and as it proceeds, it will keep searching for other good options while validating the optimality of the current “best”.

```python
    def UCTSelectChild(self, C = sqrt(2)):
        """ Use the UCB1 formula to select a child node. C in the equation below is a bias parameter
            to adjust for the algorithm's tradeoff between exploration and exploitation. """
        s = sorted(self.childNodes, key = lambda c: c.score/c.visits + C * sqrt(log(self.visits)/c.visits)) 
        return s[-1] # Return the child node with the largest value
```

### Node Expansion

In this step, a new node is added to the tree as a child node of the node selected in the previous step. Only a single node is newly introduced to the tree at every iteration. In the figure, the expanded node has the numbers 0/0 because it has neither observed any wins (or scores) nor simulations.

```python
        # Expand
        if node.untried_moves!=[]:                  # If we can expand (i.e. state/node is non-terminal)
            move = random.choice(node.untried_moves)
            state.do_move(move)
            node = node.add_child(move, state)      # Add child and descend tree
```

### Simulation / Rollout

The simulation step consists of randomly choosing moves until the algorithm reaches the terminal state or a specified threshold. Once the terminal conditions are met, then the algorithm calculates and returns a result of how well it performed as a score. This score is then passed to the backpropagation phase. This stage relies on a forward model that provides us with the outcomes of an action in any state.

```python
        # Rollout
        search_depth_count = 0                      # Reset counter
        # Repeat while the state is non-terminal and the n_search_depth limit isn't yet reached:
        while state.game_state()=='not over' and search_depth_count<=n_search_depth:
            state.do_move(random.choice(state.get_moves()))
            search_depth_count += 1
```

### Backpropagation

Once the value of the newly introduced node is determined in the simulation phase, the tree structure is updated. In the backpropagation step, the algorithm updates the perceived value of a given state, not just to the state it executed in the simulation but also every state that led to that state in the tree. The collection of the updated nodes can be observed by tracing the arrow that leads back up to the original parent node in the figure above. This updating scheme allows the algorithm to search for early actions that may lead to opportunities that that may be observed in the future. The scores are updated until the root node (starting point) is reached.

Through the above four stages, we can take decisions to a fixed point in the tree, simulate their outcome, propagate back the perceived value of it. This process is repeated multiple times to balance out the optimal set of actions. Once the simulation count limit is reached, the algorithm chooses the optimal action leading to the state with the highest value.

```python
        # Backpropagate
        while node is not None: # Backprop from the expanded node and work back to the root node.
            node.update(state.get_result()) # State is terminal. Update node with the result
            node = node.parent_node
```

---------------------------

## Why MCTS for 2048?

The 2048 game follows a Markov decision process. Specifically, it is a discrete-time and discrete-action stochastic control problem.

- The action space is discrete (up, down, left, right), and the number of available actions at each stage is small (<= 4)
- Timesteps are discrete and well-defined.
- After each action is taken, there is a stochastic (random) process, which is easily modeled using multiple Monte Carlo simulations/sampling.

The above properties make MCTS an appropriate choice for the 2048 game.

Note that the MCTS is not the only algorithm that is suitable for solving 2048. Minimax, for instance, is a noteworthy alternative. But here we only deal with MCTS.

------------------

## MCTS Hyperparameters

The main hyperparameters of the MCTS algorithm are: `nSearchPath`, `nSearchDepth`, and `explorationConst`, specifying the number of search paths, the depth of search, and the exploration constant. These hyperparameters together determine the tradeoff between exploration and exploitation.

For this work, I have used the values below, and verified that the combination of values gives adequate results. These values can be modified to further optimize for the performance (i.e. the rate of success).

| nSearchPath | nSearchDepth | Exploration Const. |
| ----------- | ------------ | ------------------ |
| 50          | 5            | 100                |

-----------------

## Results

Running an instance of `MCTS.py` script will initialize a MCTS algorithm and attempt to solve a game of 2048. Below is an example of a successful run where the algorithm was able to reach 2048 before terminating.

<center><img src="https://github.com/poomstas/2048_MCTS/raw/main/readme_img/Z_WonGame.png" alt="Won Game" style="max-width:80%;" /></center>

---------------

## Observations

- I could probably do just as well with fewer search paths and depths at the beginning. When a new game begins and there are many empty cells in the grid, there is almost always no action that would either make the player lose the game, or put him at a significant disadvantage in the subsequent states. It is only when the majority of the grid cells are filled up that choosing an optimal action starts to become more critical. Adjusting the algorithm for this observation would make it more computationally efficient. It may even be possible to simply sample randomly from the available four (or fewer) actions until a certain fraction of grid cells are occupied, after which the algorithm would compute MCTS with adjusted search hyperparameter values.
- However, when there are many empty cells because the game has recently reached a high number (e.g. 1024), then the player needs to be more cautious in selecting the next set of actions. The above observation only applies at the very beginning of the game.
- In a similar sense, I can adaptively adjust (i.e. increase) the number of search paths as the game progresses. This is also because I can computationally afford to search a lot more in the later stages because each simulation doesn't run as long until the game reaches terminal state (expand and rollout stages).
- It is interesting that when I play the game, I tend to approach it more like a Markov model than this algorithm does, i.e. only considering the next one or two steps. I tend to resort to simple heuristics than to simulate multiple alternatives in my head to estimate the optimal action.

------------------------

## References

Cameron B. Browne, Edward Powley, Daniel Whitehouse, Simon M. Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, Simon Colton (2012). A Survey of Monte Carlo Tree Search Methods. *IEEE TRANSACTIONS ON COMPUTATIONAL INTELLIGENCE AND AI IN GAMES*, *4*(1), 1–43. https://doi.org/10.1109/TCIAIG.2012.2186810

David Silver, Aja Huang, Chris J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, Demis Hassabis (2016). Mastering the game of Go with deep neural networks and tree search. *Nature*, *529*(7585), 484–489. https://doi.org/10.1038/nature16961