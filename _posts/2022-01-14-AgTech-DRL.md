---
layout: post
title:  "Deep Reinforcement Learning + AgTech"
date:   2022-01-14
excerpt: "Optimizing growth conditions using a crop simulator and Deep Reinforcement Learning (DRL)"
image: "/images/Agriculture4.jpg"
tags:
    - reinforcementlearning
    - agtech
---

### Problem Statement and Overview

The objective of this project is to find the optimum combination of 13 continuous control (action) variables through time to maximize wheat's crop yield. The crop yield is estimated using the `PCSE-v0` crop simulator provided in [this link](https://github.com/poomstas/spwk-agtech-task.git). 

- The crop cultivation simulation model is provided as a Python gym environment.
- The reinforcement learning task is episodic, each of which continues until the `DVS` variable’s (numerical representation of developmental stage) value reaches 2, or the simulation is complete.
- The model follows a Markov decision process (MDP) framework, with 11 observation (state) variables, and 13 continuous action variables. 

Further information regarding the observation and control variables is provided in the tables below.



**Table: Observation Variables**

| Variable Name | Variable                                | Min    | Max    | Unit                          |
| ------------- | --------------------------------------- | ------ | ------ | ----------------------------- |
| DVS           | Development Stage                       | 0      | 2      | Stage                         |
| LAI           | Leaf Area Index                         | 0      | 10     | ha/ha                         |
| TAGP          | Total Above Ground Production           | 105    | 30,000 | kg/ha                         |
| TWSO          | Total Dry Weight of Storage Organs      | 0      | 11,000 | kg/ha                         |
| TWLV          | Total Dry Weight of Leaves              | 68.25  | 7,500  | kg/ha                         |
| TWST          | Total Dry Weight of Stems               | 36.75  | 12,500 | kg/ha                         |
| TWRT          | Total Dry Weight of Roots               | 105    | 4,500  | kg/ha                         |
| TRA           | Crop Transpiration Rate                 | 0      | 2      | cm/day                        |
| RD            | Rooting Depth                           | 10     | 120    | cm                            |
| SM            | Soil Moisture                           | 0.3    | 0.57   | cm<sup>3</sup>/cm<sup>3</sup> |
| WWLOW         | Total Amnt of Water in the Soil Profile | 54.177 | 68.5   | cm                            |



**Table: Control Variables (Continuous)**

| Variable Name | Variable                                                     | Min                    | Max                     | Unit                |
| ------------- | ------------------------------------------------------------ | ---------------------- | ----------------------- | ------------------- |
| IRRAD         | Incoming Global Radiation                                    | 0                      | 4.0 × 10<sup>7</sup>    | J/m<sup>2</sup>/day |
| TMIN          | Daily Min Temp                                               | -50                    | 60                      | Celsius             |
| TMAX          | Daily Max Temp                                               | -50                    | 60                      | Celsius             |
| VAP           | Daily Mean Vapor Pressure                                    | 0.06 × 10<sup>-5</sup> | 199.3 × 10<sup>-4</sup> | hPa                 |
| RAIN          | Daily Total Rainfall                                         | 0                      | 25                      | cm/day              |
| E0            | Penman Potential Evaporation from a Free Water Surface       | 0                      | 2.5                     | cm/day              |
| ES0           | Penman Potential Evaporation from Moist Bare Soil Surface    | 0                      | 2.5                     | cm/day              |
| ET0           | Penman or Penman-Monteith Potential Evaporation for a Reference Crop Canopy | 0                      | 2.5                     | cm/day              |
| WIND          | Daily Mean Wind Speed at 2m Height                           | 0                      | 100                     | m/sec               |
| IRRIGATE      | Amnt of Irrigation in cm water applied on this day           | 0                      | 50                      | cm                  |
| N             | Amnt of N fertilizer in kg/ha applied on this day            | 0                      | 100                     | kg/ha               |
| P             | Amnt of P fertilizer in kg/ha applied on this day            | 0                      | 100                     | kg/ha               |
| K             | Amnt of K fertilizer in kg/ha applied on this day            | 0                      | 100                     | kg/ha               |



In the given gym environment, the above action variables are scaled from -1 to 1 using the minimum and maximum values for computational convenience.

The ultimate objective of the challenge is to create an agent that 

1. maximizes the net profit, 
2. maintains high training stability, and 
3. achieves fast convergence.

The report continues with the Executive Summary section where I make the final recommendation. Then the three attempts are summarized in the order they were conducted, followed by the Conclusion and Future Works sections at the end.

---

This repo contains an application of the a few deep reinforcement learning techniques to a plant growth simulator to optimize the growth conditions to maximize plant productivity.

This work uses a crop cultivation simulation model provided as a python gym environment. 

The crop to be studied is wheat, and the variables involved can be largely separated into two groups: observation variables, and action variables. 

The model follows a markov decision process (MDP) frameowrk, with 11 observation (state) variables, and 13 continuous action variables. Further details regarding these variables are provided in the table below. 

---



### Executive Summary

Three reinforcement learning techniques were used to maximize the total episodic reward: DDPG, TD3 and SAC. I have selected these methods primarily because they:

1. can be applied to problems wherein the action variables are continuous,

2. achieve high sampling efficiency through replay buffers, and

3. have proved to be successful (i.e. high stability & fast convergence) in other similar environments.

Further discussions on the inner workings and the implementational details of the algorithms are provided in the respective sections of the report.

As summarized in the table below, the SAC yielded the best results, and was selected to be the final submission for the challenge. This section summarizes the performance of the trained SAC model.



**Table: Max Total Episodic Reward for Three Algorithms**

| DRL Algorithm | Max Total Episodic Reward ($/ha) |
| ------------- | -------------------------------- |
| DDPG          | 0                                |
| TD3           | 1,406                            |
| SAC           | 2,802                            |

Because the SAC algorithm is inherently stochastic, it gives different results every time it is run. To evaluate the trained model accurately, I executed the SAC algorithm 1,000 times to visualize the total reward distribution. The resulting graph and data are provided below:



**Table: Average and Maximum Total Episodic Reward for SAC**

| Average Total Episodic Reward ($/ha) | Max Total Episodic Reward ($/ha) |
| ------------------------------------ | -------------------------------- |
| 2250.47                              | 2802.47                          |



**Figure: Distribution of Total Episodic Rewards Retrieved from 1,000 Episodes**

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/A.png" style="max-width:100%;" /></center>



The hyperparameters used to train the SAC model is summarized in the table below:

**Table: Hyperparameters Used to Train the Final Selected Model (SAC)**

| Hyperparameter                  | Value          |
| ------------------------------- | -------------- |
| Optimizer                       | Adam           |
| alpha (learning rate for actor) | 0.001          |
| beta (learning rate for critic) | 0.001          |
| Discount Factor                 | 0.99           |
| tau                             | 0.01           |
| Reward Scale                    | 18             |
| Batch Size                      | 100            |
| Replay Buffer Size              | 10<sup>6</sup> |
| Layer 1 Size                    | 256            |
| Layer 2 Size                    | 256            |
| Max Timesteps Per Episode       | 50,000         |

The subsequent sections details the attempts in the order they were made.

## Trial #1: Deep Deterministic Policy Gradient (DDPG)



### Algorithm

The first algorithm used was Deep Deterministic Policy Gradient (DDPG). DDPG is a deep reinforcement learning technique that draws from both Q-learning and policy gradients. One of the motives for creating DDPG was that Deep Q-Network (DQN) could only handle cases where the action spaces were discrete and low-dimensional. This was the primary basis for selecting DDPG as my first attempt to solve the problem, which involves a continuous action variable.

DDPG learns a Q-function and a policy simultaneously, and uses off-policy data and the Bellman equation to modify the Q-function. It then uses the Q-function to update the policy (Lillicrap et al., 2016). Simply put, DDPG is an approach that attempts to solve one of the major limitations of DQN (i.e. the requirement that the action space is discrete and low-dimensional). DDPG simultaneously draws from the successes of DQN by implementing two of its ideas: the replay buffer and the target network.

DDPG overcomes the above limitation by taking advantage of the fact that when the action space is continuous, the (optimal) action-value function is differentiable with respect to the action variable. Using this, a gradient-based learning rule for a policy can be constructed, as below.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/AB.png" alt="DDPG Main Equation" style="max-width:100%;" /></center>


The gradient values are then used to update the Q-function and the policy. Here, soft-updating is used to ensure that the updating procedure retains some stability.

The overview of the DDPG algorithm in the form of pseudocode is provided below.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/B.png" alt="DDPG Algorithm Pseudocode" style="max-width:100%;" /></center>



### Implementation & Results

#### Plateau Detection

To ensure that the training algorithm (applied to all three algorithms) does not continue running indefinitely, I have implemented a simple plateau detection. The algorithm calculates the mean reward values of the most recent n (default set at 100) values and the most recent 2n values. If the difference is less than 0.1%, then the training is assumed to have reached a plateau, and is terminated. The implementation can be found in a function called `has_plateaued`.


#### Grid-Based Hyperparameter Search

Several references have mentioned that DDPG is known to be sensitive to hyperparameters (Duan et al., 2016 and Henderson et al., 2017), and accordingly, a hyperparameter search algorithm was implemented. The configuration for the grid-based hyperparameter search is summarized in the table below.



**Table: Grid-Based Hyperparameter Search Configuration for DDPG**

| Hyperparameter                                 | Min      | Max     | Increment |
| ---------------------------------------------- | -------- | ------- | --------- |
| alpha (learning rate for actor)                | 0.000025 | 0.00025 | × 10      |
| beta (learning rate for critic)                | 0.000025 | 0.00025 | × 10      |
| Discount Factor                                | 0.99     | 0.99    | -         |
| tau (controls soft updating of target network) | 0.0001   | 0.001   | × 10      |
| Batch Size                                     | 64       | 64      | -         |
| Layer 1 Size                                   | 400      | 400     | -         |
| Layer 2 Size                                   | 300      | 300     | -         |



The graphs below show the training results using the above hyperparameter set configurations. 

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/C.png" alt="" style="max-width:100%;" /></center>



`last_100_reward_avg` is the moving window average of the last 100 rewards. It is used to smooth out the noise of the graph above.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/D.png" alt="" style="max-width:100%;" /></center>

The results are subpar. With the exception of very few spikes in the `episode_reward`, none of the episode runs have achieved net positive return/profit. According to Fujimoto et al. (2018), there is inherently an overestimation bias in Actor-Critic methods where the policy is updated using a deterministic policy gradient, leading to biased--and thus suboptimal--policies. 

At this stage, I have determined that it may be more time-efficient to try out an improved algorithm than to try to find higher-performing hyperparameter sets.


## Trial # 2: Twin-Delayed Deep Deterministic Policy Gradient (TD3)

### Algorithm

Twin-Delayed Deep Deterministic Policy Gradient (TD3) is an off-policy algorithm that is also designed to be used for environments with continuous action spaces. TD3 attempts to improve upon the problem of overestimation bias in the DDPG algorithm. It does so by learning two Q-functions (as opposed to one, as in DDPG), and uses the smaller of the two Q-values to use as targets in calculating the loss functions. 

Another noteworthy aspect of TD3 is that TD3 adds noise to the target action. This is done to exploit the Q-function errors by smoothing out Q along with  changes in action.

Using the above modifications have effectively addressed the value overestimation problem, the efficacy of which has been demonstrated by applying TD3 and other algorithms on standard OpenAI gym environments (Fujimoto et al., 2018).



**Table: Max Average Return over 10 trials of 1 million time steps (Fujimoto et al., 2018)**

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/E.png" alt="" style="max-width:100%;" /></center>



<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/F.png" alt="" style="max-width:50%;" /></center>



### Implementation & Results

In a fashion similar to how DDPG was approached, plateau detection and grid-based hyperparameter search was conducted. The grid-based hyperparameter search configuration is summarized in the table below.



**Table: Grid-Based Hyperparameter Search Configuration for TD3**

| Hyperparameter                                 | Min    | Max   | Increment |
| ---------------------------------------------- | ------ | ----- | --------- |
| alpha (learning rate for actor)                | 0.0001 | 0.001 | × 10      |
| beta (learning rate for critic)                | 0.001  | 0.1   | × 10      |
| Discount Factor                                | 0.99   | 0.99  | -         |
| tau (controls soft updating of target network) | 0.0005 | 0.005 | × 10      |
| Batch Size                                     | 100    | 300   | 100       |
| Layer 1 Size                                   | 400    | 400   | -         |
| Layer 2 Size                                   | 300    | 300   | -         |
| Max No. of Episodes per Train                  | 10000  | 10000 | -         |

Below are the results of training using varying sets of hyperparameters. The number of curves on the graph below coincide with that of the combinations of hyperparameters selected for this study. Each dot on the curve represents a completed episode of the simulation run.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/G.png" alt="" style="max-width:100%;" /></center>

`best_reward_so_far` graph visualizes the maximum episodic total reward gained from an episode as the SAC algorithm proceeds with the training. As expected, the graphs are monotonically increasing.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/H.png" alt="" style="max-width:100%;" /></center>

Zooming into the positive rewards, we have the following graph. The maximum reward we have here is **$1406/ha**. The instance that gave the result is 

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/I.png" alt="" style="max-width:100%;" /></center>

The graph below shows a smoothed version of the `Reward` graph shown above. The smoothing was done using an average moving window of size 100. 

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/J.png" alt="" style="max-width:100%;" /></center>


Securing a positive reward, which I was not able to with DDPG, is an improvement. However, at this stage I learned that there is an algorithm called SAC that takes on a more general form than TD3. I decided that if TD3 is a special case of SAC, then it would be more time-efficient to include TD3’s results while conducting the hyperparameter search for SAC. 

Given that the TD3 experiments above are conducted rather sparsely, I suspect (in retrospect) that TD3’s results may benefit from additional hyperparameter tuning in a search density similar to or greater than that of SAC.


## Trial # 3: Soft Actor-Critic (SAC)

### Algorithm

The Soft Actor-Critic (SAC) model takes a more general form than TD3. In fact, removing stochasticity from SAC gives a model formulation identical to that of TD3.

SAC is an algorithm that optimizes a stochastic policy.  One of SAC’s distinct features is that it employs the maximum entropy concept to modify the RL objective so that the actor attempts to maximize both the expected reward and the entropy. As a result, it brings a significant improvement to exploration and robustness (Haarnoja et al., 2018). Practically, the maximum entropy concept significantly reduces the algorithm’s sensitivity to hyperparameters, making the hyperparameter tuning process much easier than that of DDPG and TD3.

Other notable features of SAC is that it 1) takes on an actor-critic structure with separate policy and value function networks, and 2) is an off-policy formulation that enables reuse of previously collected data for efficiency (Haarnoja et al., 2018).

The SAC algorithm’s pseudocode is provided below for reference.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/K.png" alt="" style="max-width:50%;" /></center>


### Implementation & Results

Below are the results of training using varying sets of hyperparameters. The number of curves on the graph below coincide with that of the combinations of hyperparameters selected for this study. Each dot on the curve represents a completed episode of the simulation run.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/L.png" alt="" style="max-width:100%;" /></center>


`best_reward_so_far` graph visualizes the maximum episodic total reward gained from an episode as the SAC algorithm proceeds with the training. As expected, the graphs are monotonically increasing.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/M.png" alt="" style="max-width:100%;" /></center>

Maximum one-time reward value from ‘best_reward_so_far’ graph is $2862/ha, and the unique identifier/name for the trial is:

```
SAC_PCSE_alpha_0.001_beta_0.001_tau_0.01_RewScale_18_batchsize_100_layer1size_256_layer2size_256_nGames_50000_patience_1000_20211023_005715_Run commands auto-generated 20211016
```

The above instance is selected as the final result for this project.

`last_100_reward_avg` is the moving window average of the last 100 rewards.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/N.png" alt="" style="max-width:100%;" /></center>

Below is a graph that helps visualize the key hyperparameters’ influence on one of the key metrics, `best_reward`. SAC has two core hyperparameters that need to be carefully tuned: tau, which determines the rate at which the target networks (both actor and critic) are updated, and reward scale, which acts as the temperature of the energy-based optimal policy, and thus determines the extent of stochasticity of SAC.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/O.png" style="zoom:50%;" /></center>


The figure above is color-coded according to the highest reward observed during training. We can visually filter out the best-performing cases to see if a pattern exists among tau and reward scale values, as below.

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/P.png" style="zoom:50%;" /></center>


It seems that the optimal reward scale value is approximately 13 to 19, while the tau value varies over a wider range. Taking the hyperparameter set that had the highest single-episode total reward (profit) value, we have below:

**Hyperparameters Used to Train the Final Selected Model (SAC)**

| Hyperparameter                     | Value  |
| ---------------------------------- | ------ |
| Optimizer                          | Adam   |
| alpha (learning rate for actor)    | 0.001  |
| beta (learning rate for critic)    | 0.001  |
| Discount Factor                    | 0.99   |
| tau (target smoothing coefficient) | 0.01   |
| Reward Scale                       | 18     |
| Batch Size                         | 100    |
| Replay Buffer Size                 | 106    |
| Layer 1 Size                       | 256    |
| Layer 2 Size                       | 256    |
| Max Timesteps Per Episode          | 50,000 |



**Environment Render of Best-Performing Actions (Profit: $2802.47/ha)**

<center><img src="https://github.com/poomstas/AgTech_DRL/raw/main/README_Figures/Q.png" alt="" style="max-width:100%;" /></center>


The above results can be reproduced by running the `check_best_performing_action.py` script. 

## Conclusion

In this work, I was find the set of control variables for the PCSE crop simulation that maximizes the net profit. Three algorithms (DDPG, TD3, and SAC) were employed to solve the problem, and the results were compared to determine which approach yields the best results. Among the three, the SAC gave the most profitable result, with `$2802/ha`, and is selected as the final recommendation. SAC algorithm’s use of stochastic policies, off-policy formulation, replay buffer, and entropy regularization have allowed a stable and sample-efficient training, which resulted in the best final result.


## Future Works

- In the above work, more time was spent trying out hyperparameter sets for SAC than DDPG and TD3. Although the SAC is known to be more efficient (Haarnoja et al., 2018), DDPG and TD3 may yield better results (compared to what was observed) with the right hyperparameters, and may be worth an investigation. 
- There are a number of notable developments since SAC, such as transformer-based models. It should be interesting to learn more about the newly developed models and compare their performances against the ones studied in this work.
- Use a more advanced method for hyperparameter search in the place of grid-based approach. Potential candidates are: latin hypercube sampling, Bayesian optimization, infinite-armed bandit-based approach, etc.
- Because one of the objectives is to find the set of actions that yield the best reward, it may be helpful to keep track of the best reward and save the corresponding action set while the training is proceeding. In a few cases I seem to have lost some high-performing models ($2800+/ha) due to the absence of a timely saving feature.

## References

- Haarnoja T., Zhou, A., Abbeel P., & Levine S. (2018) Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.
- Duan, Y., Chen, X., Houthooft, R., Schulman, J., and Abbeel, P. (2016) Benchmarking deep reinforcement learning for continuous control. In *International Conference on Machine Learning (ICML)*
- Henderson, P., Islam, R., Bachman, R., Pineau, J., Precup, D., and Meger, D. (2017) Deep reinforcement learning that matters.
- Fujimoto, S., van Hoof, H., Meger, D. (2018) Addressing Function Approximation Error in Actor-Critic Methods.
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Eerez, T., Tassa, Y., Silver, D., and Wierstra, D. (2016) Continuous Control with Deep Reinforcement Learning.



------------------

## Running the Scripts

Check out the GitHub repo here: [github.com/poomstas/AgTech_DRL](github.com/poomstas/AgTech_DRL)

### Create a Conda Environment

```
conda create --name AgTech_DRL python=3.8
conda activate AgTech_DRL
pip install git+https://github.com/poomstas/spwk-agtech-task.git
python -m spwk_agtech.make_weather_cache
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c conda-forge tensorboard
conda install matplotlib
```



### Running the Training Scripts

#### The main training scripts in each folder begin with `main`. 

`main_train_ddpg.py` for DDPG

`main_train_td3.py` for TD3

`main_train_sac.py` for SAC


#### Specifying Hyperparameters

When running the training scripts, hyperparameters can be specified. For instance:

`python main_train_ddpg.py --alpha 0.01 --beta 0.1 --tau 0.01 --gamma 0.95 --batch_size 32 --layer1_size 300 --layer2_size 200`

To see which hyperparameters can be specified, run: 

`python main_train_ddpg.py --help`



### Check Best-Performing Action Set

To visualize the results of the best-performing action set (acquired by training an SAC model), run:

`python check_best_performing_action.py`



### Load the Best Trained SAC Model and Test

To load the best-case SAC model, run multiple episodes on the given environment and calculate the average and maximum episode rewards,
run the following:

`python test_SAC_model.py`

You can adjust the total number of episodes to run in the script by adjusting the `N_TEST_CASE` variable in the script.
