import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class BeerGameEnv(gym.Env):
    """ OpenAI Gym Environment for the MIT Beer Game
    """
    #metadata = {'render.modes': ['human']}

    def __init__(self, env_init_info={}):
        '''Initial inventory is a list with the initial inventory position for each level
        '''
        self.DEBUG = False
        # Valores padrões do MIT Beer Game (usados se não forem passados outros valores)
        beer_game_std_levels      = 4
        beer_game_std_demands     = [4]*4 + [8]*31
        beer_game_std_inventory   = 12 + np.zeros(beer_game_std_levels)
        beer_game_std_ship_delay  = 2
        beer_game_std_ship_value  = 4
        beer_game_std_orders_value= 4
        beer_game_std_inv_cost    = 1
        beer_game_std_backlog_cost= 2

        # Número de níveis da cadeia
        self.levels       = env_init_info.get('levels', beer_game_std_levels)
        # Custo por unidade em estoque por semana
        self.inv_cost     = env_init_info.get('inv_cost', beer_game_std_inv_cost)
        # Custo por unidade em backlog por semana
        self.backlog_cost = env_init_info.get('backlog_cost', beer_game_std_backlog_cost)

        # Demanda dos clientes a cada semana
        self.customer_demand = np.asarray(env_init_info.get('customer_demand', beer_game_std_demands), dtype=int)
        # Quantidade inicial de estoque em cada nível
        self.initial_inventory = np.asarray(env_init_info.get('initial_inventory', beer_game_std_inventory), dtype=int)
        # Número de semanas a simular
        self.max_weeks = len(self.customer_demand)
        # Máximo leadtime de entrega
        self.shipment_delays = np.asarray([beer_game_std_ship_delay] + env_init_info.get('shipment_delays', [beer_game_std_ship_delay]*self.max_weeks))
        # Valor inicial de ítens em transporte
        self.initial_shipment_value = env_init_info.get('initial_shipment_value', beer_game_std_ship_value)
        # Pedidos colocados inicialmente
        self.initial_orders_value = env_init_info.get('initial_orders_value', beer_game_std_orders_value)

        # Estrutura para guardar todas as entregas. Por tempo, por nível.
        max_shipment_week = self.max_weeks+1
        for i in range(self.max_weeks+1):
            if i+self.shipment_delays[i]+1 > max_shipment_week:
                max_shipment_week = i+self.shipment_delays[i]+1
        self.initial_shipment = np.zeros((max_shipment_week+1, self.levels), dtype=int)
        # Tratando as entregas pendentes já no momento inicial
        self.initial_shipment[1:1+self.shipment_delays[0]][:] = self.initial_shipment_value

        # Pedidos colocados para o nível acima
        self.initial_orders_placed = self.initial_orders_value + np.zeros(self.levels, dtype=int)
        # Pedidos pendentes que chegaram do nível abaixo (posição zero é a demanda do cliente,
        # e valor colocado aqui é ignorado depois)
        self.initial_incoming_orders = self.initial_orders_value + np.zeros(self.levels, dtype=int)

        self.current_state = None

        # Tratando variáveis do OpenAI Gym environment (# Ver uso do MultiDiscrete)
        #self.action_space = spaces.Discrete(self.action_codes**self.levels)
        #self.observation_space = spaces.Discrete(self.state_codes**self.levels)

    def step(self, action):
        self.week += 1

        # 1. Receive inventory and advance shipment delay

        # Os estoques recebem o que estava previsto nas entregas
        self.inventory += self.shipments[self.week][:]
        # Não é necessário "avançar" as entregas, pois estamos guardando todas
        # no vetor (por tempo)

        # 2. Fill orders

        # Retailer obtém a demanda do cliente
        self.incoming_orders[0] = self.customer_demand[self.week-1]
        # Demais níveis recebem demanda do nível inferior (# 4. Advance the order slips)
        self.incoming_orders[1:] = self.orders_placed[:-1]

        # Os pedidos a serem tratados são o que veio dos níveis abaixo da cadeia
        # mais o que tinha pendente de backlog
        orders_to_fill = self.incoming_orders + self.backlog

        # O quanto será enviado será o total de pedidos a serem tratados, se
        # possível, ou tudo que tem, se não for possível.
        orders_to_deliver = np.minimum(self.inventory, orders_to_fill)

        # Já a entrega considera apenas o que realmente pode ser entregue.
        # Se o tempo de delay é zero, entrega direto nos estoques dos níveis abaixo
        if self.shipment_delays[self.week] == 0:
            self.inventory[:-1] += orders_to_deliver[1:] # A primeira posição é para o cliente
        else: # Se delay é maior que zero, agenda a entrega
            self.shipments[self.week+self.shipment_delays[self.week]][:-1] += orders_to_deliver[1:]

        # 3. Record the invetory or backlog

        # Desconta do estoque tudo que foi enviado para entrega
        self.inventory -= orders_to_deliver
        # Guarda como backlog tudo que não foi possível entregar
        self.backlog = orders_to_fill - orders_to_deliver

        # 4. Advance the order slips

        # Movendo os pedidos colocados para os pedidos chegando
        # Já feito no Passo 2

        # Pedidos para a fábrica (último nível) são colocados para entrega
        if self.shipment_delays[self.week] == 0:
            self.inventory[-1] += self.orders_placed[-1]
        else:
            self.shipments[self.week+self.shipment_delays[self.week]][-1] += self.orders_placed[-1]

        # 5. Place orders

        # Cada nível passa para o nível acima um pedido de tamanho X+Y
        # onde X é o que recebeu de demanda e Y é a quantidade a decidir pelo agente

        self.orders_placed = self.incoming_orders + action

        self.all_orders_placed[:,self.week] = self.orders_placed.T

        # 6. Tratando agora as questões de Aprendizado (recompensa e próximo estado)

        self.current_state = self._observation()

        # A recompensa é a soma ponderada dos custos de estoque e backlog
        reward = -np.sum(self.inv_cost*self.inventory + self.backlog_cost*self.backlog)
        self.inventory_costs += self.inv_cost*self.inventory
        self.backlog_costs += self.backlog_cost*self.backlog

        is_terminal = self.week == self.max_weeks

        if self.DEBUG: print('env.step()', self.inventory - self.backlog, self.current_state)

        return self.current_state, reward, is_terminal, {}

    def reset(self):
        self.week = 0
        self.inventory = np.copy(self.initial_inventory)
        self.backlog = np.zeros(self.levels, dtype=int)
        self.orders_placed = np.copy(self.initial_orders_placed)
        self.incoming_orders = np.copy(self.initial_incoming_orders)
        self.shipments = np.copy(self.initial_shipment)

        self.inventory_costs = np.zeros(self.levels, dtype=int)
        self.backlog_costs = np.zeros(self.levels, dtype=int)

        self.all_orders_placed = np.zeros((self.levels,self.max_weeks+1), dtype=int)
        self.all_orders_placed[:,0] = self.initial_orders_value

        self.current_state = self._observation()

        return self.current_state

    def render(self, mode='human'):
        print('\n' + '='*20)
        print('Week:\t', self.week)
        print('Inventory:\t', self.inventory, self.backlog, self.inventory - self.backlog)
        print('Incoming order:\t', self.incoming_orders)
        print('Orders placed:\t', self.orders_placed)
        if self.week < self.max_weeks:
            print('Next customer demand:\t', self.customer_demand[self.week])
        #print('Print shipments:\n', self.shipments)
        #print('self.shipments[', self.week+1, ':', self.week+self.shipment_delays[self.week]+1, ']')
        print('Next shipments:\t', [(i,list(self.shipments[i])) for i in range(self.week+1, self.week+6) if i < len(self.shipments)])
        print('Current delay:\t', self.shipment_delays[self.week])
        print('Inventory costs:\t', self.inventory_costs)
        print('Backlog costs:\t', self.backlog_costs)
        #if (self.week == self.max_weeks):
        #    print('All orders placed:')
        #    for level in range(self.levels):
        #        print(level, self.all_orders_placed[level])

    def close(self):
        pass

    def _observation(self):
        return self.inventory - self.backlog
