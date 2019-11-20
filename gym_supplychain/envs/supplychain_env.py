import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class SupplyChainEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    def array_to_int(self, array, num_codes):
        code = 0
        for i in range(len(array)):
            code += array[i]*(num_codes**i)
        return code

    def int_to_array(self, code, num_levels, num_codes):
        array = [0]*num_levels
        for i in range(num_levels-1,0,-1):
            array[i] = code // num_codes**i
            code = code % num_codes**i
        array[0] = code
        return array

    def observation(self, state):
        return self.array_to_int(state, self.state_codes)

    def decode_actions(self, action):
        return self.int_to_array(action, self.levels, self.action_codes)

    def code_state(self, inventory):
        state = [0]*len(inventory)
        for i in range(len(inventory)):
            if inventory[i] < -6:
                state[i] = 0
            elif inventory[i] < -3:
                state[i] = 1
            elif inventory[i] < 0:
                state[i] = 2
            elif inventory[i] < 3:
                state[i] = 3
            elif inventory[i] < 6:
                state[i] = 4
            elif inventory[i] < 10:
                state[i] = 5
            elif inventory[i] < 15:
                state[i] = 6
            elif inventory[i] < 20:
                state[i] = 7
            else:
                state[i] = 8
        return np.asarray(state)


    def __init__(self, env_init_info={}):
        '''Initial inventory is a list with the initial inventory position for each level
        '''

        # Número de níveis da cadeia
        self.levels = 4
        self.state_codes = 9
        self.action_codes = 4
        self.inv_cost = 1
        self.backlog_cost = 2
        self.initial_ditr = 12 + np.zeros(self.levels+1, dtype=int)

        self.action_space = spaces.Discrete(self.action_codes**self.levels)
        self.observation_space = spaces.Discrete(self.state_codes**self.levels)

        # Quantidade de itens em cada estoque
        self.initial_inventory = np.asarray(env_init_info['initial_inventory'])
        self.customer_dem = env_init_info['customer_dem']
        #self.lead_times = env_init_info['lead_times']
        self.max_n = len(self.customer_dem)
        print('max_n', self.max_n)

        self.current_state = None
        # Ver uso do MultiDiscrete


    def step(self, action):
        Y_actions = np.asarray(self.decode_actions(action))
        #Y_actions = np.asarray(self.decode_actions(0))

        # o último nível recebe a demanda do cliente
        self.A_orders[-1] = self.customer_dem[self.n]

        # 1. Chegada de pedidos encomendados (de acordo com o leadtime)
        #print('ord', self.A_orders)
        #print('inv', self.S_inventory)
        #print('dis', self.T_distr)
        #print()

        self.S_inventory += self.T_distr[:-1]


        # 2. Recebimento de novos pedidos


        # 3. Atendimento dos novos pedidos recebidos

        self.T_distr[0] = self.A_orders[0]
        # A quantidade de itens disponíveis são os valores positivos do estoque
        H_inv_onhand = np.clip(self.S_inventory,0,None)
        self.T_distr[1:] = np.minimum(H_inv_onhand, self.A_orders[1:])

        self.S_inventory -= self.A_orders[1:]


        # 4. Decisão de novos pedidos

        # cada nível passa para o nível acima um pedido de tamanho X+Y
        # onde X é o que recebeu de demanda e Y é a quantidade a decidir pelo agente
        for i in range(self.levels):
            self.A_orders[i] = self.A_orders[i+1] + Y_actions[i]

        self.current_state = self.code_state(self.S_inventory)

        # A quantidade de itens disponíveis são os valores positivos do estoque
        H_inv_onhand = np.clip(self.S_inventory,0,None)
        # O backlog é a quantidade de itens que falta no estoque (valores negativos)
        C_backlog = -np.clip(self.S_inventory,None,0)
        # A recompensa é a soma ponderada dos custos de estoque e backlog
        reward = -np.sum(self.inv_cost*H_inv_onhand + self.backlog_cost*C_backlog)

        self.n += 1
        is_terminal = self.n == self.max_n

        return self.observation(self.current_state), reward, is_terminal, {}

    def reset(self):
        self.n = 0
        self.S_inventory = np.copy(self.initial_inventory)

        self.A_orders = np.zeros(self.levels+1, dtype=int)

        self.T_distr = np.copy(self.initial_ditr)

        self.current_state = self.code_state(self.S_inventory)

        return self.observation(self.current_state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass
