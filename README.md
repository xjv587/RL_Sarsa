# Implementing two advanced reinforcement learning algorithms: 

the True Online Sarsa(λ) algorithm and the REINFORCE algorithm with a baseline. Both algorithms are applied to environments from OpenAI Gym, providing hands-on experience with real-world reinforcement learning challenges.

## True Online Sarsa(λ) Algorithm: 

This part focuses on implementing the Sarsa(λ) algorithm, which is an enhancement of the traditional Sarsa method that incorporates eligibility traces for more efficient learning. The implementation involves creating a feature extractor using tiling techniques to ensure comprehensive coverage of the state space. Key considerations include managing floating-point errors and ensuring correct tiling offsets.

## REINFORCE Algorithm with Baseline: 

This section involves implementing the REINFORCE algorithm, a policy gradient method that uses a baseline to reduce variance in the gradient estimates. The implementation includes modeling the policy and value function using neural networks, with specific requirements for the optimizer and network architecture to ensure convergence. The use of PyTorch is recommended for building and training these models.
