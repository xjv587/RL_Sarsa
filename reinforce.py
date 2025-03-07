from typing import Iterable
import numpy as np
import torch
from torch.distributions import Categorical

class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here

        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_actions),
            torch.nn.Softmax(dim=-1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> int:
        # TODO: implement this method
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32)
            probs = self.model(s)
            dist = Categorical(probs)
            action = dist.sample()
        return action.item()

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        # TODO: implement this method
        self.model.train()
        s = torch.tensor(s, dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.int64)
        probs = self.model(s)
        log_prob = torch.log(probs[a])
        loss = -log_prob * gamma_t * delta
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        # TODO: implement here
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dims, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))
        self.loss_fn = torch.nn.MSELoss()

    def __call__(self,s) -> float:
        # TODO: implement this method
        self.model.eval()
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float32)
        return self.model(s).item()

    def update(self,s,G):
        # TODO: implement this method
        self.model.train()
        s = torch.tensor(s, dtype=torch.float32)
        G = torch.tensor(G, dtype=torch.float32)
        value = self.model(s)
        loss = torch.nn.functional.mse_loss(value, G)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    # TODO: implement this method
    G_0s = []
    for episode in range(num_episodes):
        s = env.reset()
        states, actions, rewards = [], [], [0]
        done = False
        while not done:
            a = pi(s)
            s_prime, r, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_prime
        states.append(s)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        G_0s.append(returns[0])
        
        for t in range(len(states)-1):
            s_t = states[t]
            a_t = actions[t]
            G_t = returns[t]
            
            baseline_value = V(s_t)
            delta = G_t - baseline_value
            
            V.update(s_t, G_t)
            pi.update(s_t, a_t, gamma**t, delta)

    return G_0s

