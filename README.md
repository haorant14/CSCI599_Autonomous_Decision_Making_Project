# Autonomous Decision-Making

## Setup

The code requires [Anaconda](https://www.anaconda.com/download).

Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:

```bash
cd code
pip install -r requirements.txt
```

## Available Maps

All available maps are provided in the folder `code/layouts` and listed in the table below.

| Map   		| File                      |
|---------------|---------------------------|
| `easy_0`      | `code/layouts/easy_0.txt` |
| `easy_1`      | `code/layouts/easy_1.txt` |
| `medium_0`    | `code/layouts/medium_0.txt` |
| `medium_1`    | `code/layouts/medium_1.txt` |
| `hard_0`      | `code/layouts/hard_0.txt` |
| `hard_1`      | `code/layouts/hard_1.txt` |

## Training

Train Dyna-Q agent using the following commands in a terminal (`map-name` is provided in the "Map"-column of the table above):

```bash
cd code
python train.py <map-name>
```
It will take a long time to train as default is 2000 episode. To Train other Agents, modify the agent on line 130 in the `train.py` accordingly.

## Evaluation

Run a greedy agent using existing Q table using following commands in terminal to validate training result over 100 episode.

```bash
python .\load_model.py <map-name> <algorithm-name> 
```

Here is an example:

```bash
python .\load_model.py hard_1 Dyna-Q
```

Here is a list of `algorithm-name`: 
- `Dyna-Q`
- `Offpolicy_MonteCarlo`
- `onpolicy_MonteCarlo`
- `Q-Learning`
- `SARSA`
- `SARSAlambda`
  

## Pre-trained Models

After training, training result(Q table) will be stored in `code/qtable` as a pickle file. you can use these for evaluation without a long time of training.

## Code Structure Overview

 `code/`contains:

- `agent.py`: different agents including Q-learning, SARSA, SARSA($\lambda$), Monte Carlo Learning, and Dyna-Q learning.
  
- `multi_armed_bandits.py.py`:different exploration/expoitation methods including Episilon Decay Greedy

- `train.py`: script for training selected agents on selected map.
  
- `load_model.py`: script for validating Q-table using a greedy agent. 
- `qtable/`: pre-trained agent Q-table for quick reference 

`figures/`: training and validating curve for different agents on different maps. 

## Hyper-parameter Tuning

To achieve the best performance, we used the following hyper-parameters: 

- Epsilon Decay Greedy: $\epsilon$ = 0.00001, decay = 0.00001
- learning rate $\alpha$ = 0.1
- discount $\gamma$  = 0.99
- eligibility trace decay factor $\lambda$ (for SARSA($\lambda$))= 0.75 
- number of planning steps(for Dyna-Q) = 50

## Results

Below shows the average discounted return over 100 episodes on different level maps for the algorithm we used. the following methods used decaying epsilon greedy as exploation/exploitation method.

| Model name         | Easy | Medium  | Hard|
| ------------------ |---------------- | -------------- |----------|
| Dyna-Q   |     0.9826         |    0.9779        |     0.9501 |
| SARSA($\lambda$)  | 0.9960  | 0.9788  | 0.8373 |
| SARSA |0.9957 | 0.9703|0.9520 |
| Q-learning|0.9845 | 0.9798 | 0.9523|
| On-Policy Monte-Carlo Learning |0.7772 | 0.6301| N/A |
| Off-Policy Monte-Carlo Learning |0.5329 | 0.3667|N/A |
