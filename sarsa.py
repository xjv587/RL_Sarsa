import numpy as np

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        self.tile_offsets = np.array([(-i / num_tilings) * tile_width for i in range(num_tilings)])
        self.feature_length = int(self.num_actions * self.num_tilings * np.prod(self.num_tiles))

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this method
        return self.feature_length
    
    def _get_tile_index(self, s, tiling_index):
        offset_s = s - self.tile_offsets[tiling_index]
        tile_indices = np.floor((offset_s - self.state_low) / self.tile_width).astype(int)
        linear_index = np.ravel_multi_index(tile_indices, self.num_tiles)
        return linear_index

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        if done:
            return np.zeros(self.feature_vector_len())
        
        feature_vector = np.zeros(self.feature_vector_len())
        for tiling_index in range(self.num_tilings):
            tile_index = self._get_tile_index(s, tiling_index)
            index = int(a * (self.num_tilings * np.prod(self.num_tiles)) + tiling_index * np.prod(self.num_tiles) + tile_index)
            feature_vector[index] = 1
        
        return feature_vector

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    #TODO: implement this function
    for episode in range(num_episode):
        done = False
        S = env.reset()
        A = epsilon_greedy_policy(S, done, w, epsilon=0.05)
        x = X(S, done, A)
        z = np.zeros_like(w)
        Q_old = 0

        while not done:
            S_prime, R, done, _ = env.step(A)
            A_prime = epsilon_greedy_policy(S_prime, done, w, epsilon=0.05)
            x_prime = X(S_prime, done, A_prime)
            Q = np.dot(w, x)
            Q_prime = np.dot(w, x_prime)

            delta = R + gamma * (not done) * Q_prime - Q
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, x)) * x
            w += alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            A = A_prime
        
    return w