import numpy as np
from cost_functions import trajectory_cost_fn
import time

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
		self.action_shape = env.action_space.shape()

	def get_action(self, state):
		""" Your code should randomly sample an action uniformly from the action space """

		# Return a default cost of -1 for the random controller
		return (np.random.uniform(low=self.action_low, high=self.action_high, shape=self.action_shape), -1)

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
		self.action_shape = self.env.action_space.shape()

		self.gen_shape = tuple([num_simulated_paths, horizon] + list(self.action_shape))


	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """

		# num_simulated_paths = K
		# horizon = H
		# action_dim = D
		# state_dim = S

		# This is K x H x D
		all_actions = np.random.uniform(low=self.action_low, high=self.action_high, shape=self.gen_shape)

		# These will be H x K x S
		all_states = []
		all_next_states = []

		curr_states = np.array([state for _ in range(self.num_simulated_paths)])
		for t in range(self.horizon):
			# This is K x D
			curr_actions = all_actions[:,t,:]
			# This is K x S
			# The dynamics model outputs the difference between states, so we must
			# add this delta to our current state to obtain the new state
			curr_next_states = curr_states + self.dyn_model.predict(curr_states, curr_actions)
			all_states.append(curr_states)
			all_next_states.append(curr_next_states)
			curr_states = curr_next_states

		costs = []

		for j in range(self.num_simulated_paths):
			# Takes (states, actions, next states)
			trajectory_states = all_states[:,j,:]
			trajectory_next_states = all_next_states[:,j,:]
			trajectory_actions = all_actions[j]
			trajectory_cost = trajectory_cost_fn(self.cost_fn, trajectory_states, trajectory_actions, trajectory_next_states)
			costs.append(trajectory_cost)

		best_trajectory = np.argmin(costs)
		best_cost = np.min(costs)

		return (all_actions[best_trajectory][0], best_cost)


