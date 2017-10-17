import numpy as np
from cost_functions import cheetah_cost_fn, trajectory_cost_fn

class Data:

	def __init__(self):
		self.states = []
		self.actions = []
		self.next_states = []

	def __len__(self):
		return len(self.states)

	def add(self, state, action, next_state):
		self.states.append(state)
		self.actions.append(action)
		self.next_states.append(next_state)

	def add_all(self, data):
		self.states += data.states
		self.actions += data.actions
		self.next_states += data.next_states

	def get_states(self):
		return np.stack(self.states, axis=0)

	def get_actions(self):
		return np.stack(self.actions, axis=0)

	def get_next_states(self):
		return np.stack(self.next_states, axis=0)

	def get_cost(self, cost_fn):
		return trajectory_cost_fn(cost_fn, self.states, self.actions, self.next_states)