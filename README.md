# Simple-A2C-using-Pytorch-MountainCarv0
## This implementation is supposed to serve as a beginner solution to the classic Mountain-car with discrete action space problem. 
- This file uses Advantage Actor critic algorithm with epsilon greedy exploration strategies. It doesn't need any open AI baseline knowledge and can be implemented using knowledge of DRL, OpenAI environment API and Pytorch. 
- The program uses a single neural network which eventually diverges into two separate heads for Actor and Critic outputs.
- The architecture of the network is Actor: 2->64->128->64->num_actions and Critic: 2->64->128->64->1 
- Epsilon greedy exploration technique is used to account for the sparse reward function(Reward -1 for each timestep untill goal state reached) of the Mountain Car. This sparse reward function makes most DRL algos useless without some reward hacking or external optimization. Its a classic problem to demonstrate exploration vs exploitation.
- Epsilon is decayed after every episode at a rate of 0.99 to allow high exploration in earlier stages and reduced exploration in subsequent stages.
- The algorithm also only backprops after intervals of few timesteps to avoid blown-up neural network weights.
- One might need to increase the number of timesteps allowed by OpenAI gym ( edit file gym->envs->__init__.py and change max_episode_steps to required value) for convergence.
- The sparse reward function makes this problem inherent. In practice,  5000 iteration episode converged after 40 episodes and 1000 iteration episode converged after 500.


