import numpy as np
from cost_functions import trajectory_cost_fn

class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        self.action_high = env.action_space.high
        self.action_low = env.action_space.low
        self.action_shape = env.action_space.shape

    def get_action(self, state):
        """ Your code should randomly sample an action uniformly from the action space """

        return np.random.uniform(low=self.action_low, high=self.action_high, size=self.action_shape)

class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self, 
                 env, 
                 dyn_model, 
                 horizon=5, 
                 cost_fn=None, 
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

        self.action_high = self.env.action_space.high
        self.action_low = self.env.action_space.low
        self.action_shape = self.env.action_space.shape

        self.gen_shape = tuple([num_simulated_paths, horizon] + list(self.action_shape))

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """

        # num_simulated_paths = K
        # horizon = H
        # action_dim = D
        # state_dim = S

        # This is K x H x D
        all_actions = np.random.uniform(low=self.action_low, high=self.action_high, size=self.gen_shape)

        # These will be H x K x S
        all_states = []
        all_next_states = []

        curr_states = np.array([state for _ in range(self.num_simulated_paths)], dtype=np.float32)
        for t in range(self.horizon):
          # This is K x D
          curr_actions = all_actions[:,t,:]
          # This is K x S
          curr_next_states = self.dyn_model.predict(curr_states, curr_actions)
          all_states.append(curr_states)
          all_next_states.append(curr_next_states)
          curr_states = curr_next_states

        all_states = np.array(all_states, dtype=np.float32)
        all_next_states = np.array(all_next_states, dtype=np.float32)

        new_acts_shape = list(all_states.shape)
        new_acts_shape[2] = self.action_shape[0]

        costs = trajectory_cost_fn(self.cost_fn, all_states, np.reshape(all_actions, tuple(new_acts_shape)), all_next_states)

        best_trajectory = np.argmin(costs)
        best_cost = np.min(costs)

        return all_actions[best_trajectory][0]


