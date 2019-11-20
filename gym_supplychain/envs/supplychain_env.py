import gym
from gym import error, spaces, utils
from gym.utils import seeding
from copy import deepcopy
import numpy as np

class SupplyChainEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def code_array(self, array, num_codes):
        code = 0
        for i in range(len(array)):
            code += array[i]*(num_codes**i)
        return code

    def decode_array(self, code, num_levels, num_codes):
        array = [0]*num_levels
        for i in range(num_levels-1,0,-1):
            array[i] = code // num_codes**i
            code = code % num_codes**i
        array[0] = code
        return array

    def observation(self, state):
        return self.code_array(state, self.state_codes)

    def decode_actions(self, action):
        return self.decode_array(action, self.levels, self.action_codes)

    def __init__(self, env_init_info={}):
        '''Initial inventory is a list with the initial inventory position for each level
        '''
        # Número de níveis da cadeia
        self.levels = len(env_init_info['initial_inventory'])
        self.state_codes = 9
        self.action_codes = 4

        self.inv_cost = 1

        self.start_state = np.asarray(env_init_info['initial_inventory'])
        self.current_state = None

        # Ver uso do MultiDiscrete

        self.action_space = spaces.Discrete(self.action_codes**self.levels)
        self.observation_space = spaces.Discrete(self.state_codes**self.levels)


    def step(self, action):
        actions = np.asarray(self.decode_actions(action))

        new_state = deepcopy(self.current_state)

        self.current_state = new_state + actions

        for i in range(len(self.current_state)):
            self.current_state[i] = min(self.state_codes-1, self.current_state[i])

        reward = np.sum(self.inv_cost*self.current_state)
        if sum(actions == 0): # penaliza ação com tudo zero
            reward -= 10

        is_terminal = False
        if np.max(self.current_state) == self.state_codes-1:
            is_terminal = True

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start_state
        return self.observation(self.current_state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
