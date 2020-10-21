import numpy as np
import heapq

import gym
from gym import spaces

from .demands_generator import generate_demand

class SC_Action:
    """ Define uma ação do ambiente da Cadeia de Suprimentos.

        As ações podem ser dos seguintes tipos:
        - SUPPLY: decide quanto fornecer de material para o próprio nó da cadeia
        - SHIP: decide quanto enviar de material para cada possível destino.

        Essa classe tem por objetivo converter um valor de ação real (entre 0 e 1)
        em um valor inteiro correspondente a ser realmente aplicado para a aquela ação.
    """
    def __init__(self, action_type, capacity=None, costs=None):
        """ Cria a ação, armazenando o tipo passado

            :param action_type: (str) Define o tipo da ação
            :param capacity: (int) Define o valor máximo da ação. Se `None` (padrão) não existe valor máximo.
        """
        # confere se o tipo de ação passado é válido
        assert action_type in ['SUPPLY', 'SHIP']

        # confere se ação de fornecimento e estoque tem capacidade válida e ação de envio tem destino válidos
        if action_type == 'SUPPLY':
            assert capacity is not None

        self.action_type = action_type
        self.capacity = capacity
        self.costs = costs

    def apply(self, action_values, max=None):
        """ Aplica o valor de ação recebido, podendo ter um limite variável.

            :param action_value: (float or list) valor (ou lista de valores) entre 0 e 1 (porcentagem) referente à ação a ser aplicada.
            :param max: (int) valor máximo da ação para essa decisão específica.
            :return: (int or list) retorna o(s) valor(es) inteiro(s) correspondente(s) à ação.
        """
        if self.action_type == 'SUPPLY':
            # No caso de fornecimento basta usar o valor percentual no máximo de
            # valor inteiro possível (mínimo entre capacidade e máximo recebido)
            if max is None:
                limit = self.capacity
            else:
                limit = min(self.capacity, max)
            supplied_amount = action_values*limit
            return supplied_amount, supplied_amount*self.costs
        elif self.action_type == 'SHIP':
            
            # Definindo a quantidade máxima de material disponível para envio
            if self.capacity is None:
                limit = max
            else:
                limit = min(self.capacity, max)
            available = limit
                        
            # Inicializando listas a serem retornadas (quantidades e custos)
            returns = [0]*len(action_values)
            costs = [0]*len(action_values)
                
            # Se tem material para ser enviado
            if available > 0:
                # formando tuplas com os valores das ações e o índice do destino
                tuples = [(action_values[i], i) for i in range(len(action_values))]
                # ordenando as tuplas pelos valores das ações
                tuples.sort()
                
                # o primeiro corte será de zero até o primeiro valor
                initial_cut = 0
                # Para cada valor e destino
                for value, idx in tuples:
                    # o corte vai até o valor da ação desse destino
                    final_cut = value 
                    # precisamos arredondar a quantidade de material pois ela é inteira
                    amount = (final_cut-initial_cut)*limit
                    # tratamento para evitar problemas de arredondamento
                    if amount > available:
                        amount = available
                    # guardando a quantidade (e o custo) referentes a o destino em questão
                    returns[idx]= amount
                    costs[idx] = amount*self.costs[idx]
                    # descontamos agora a quantidade de material disponível
                    available -= amount
                    # e o próximo corte começa onde terminou esse
                    initial_cut = final_cut

            return returns, costs
        else:
            raise NotImplementedError('Unknown action type:' + self.action_type)

    def get_type(self):
        return self.action_type

class SC_Node:
    """ Define um nó da Cadeia de Suprimentos

        Cada nó tem:
        - Um identificador.
        - Uma capacidade de estoque.
        - Uma capacidade de fornecimento.
        - Uma indicação se é nó do último nível (estágio).
        - Uma quantidade de material em estoque.
        - Uma lista de ações possíveis.
        - O número total de valores de ação esperados.
        - Uma lista de destinos.
        - Uma fila de material para chegar.
    """
    def __init__(self, label, initial_stock=0, initial_supply=None, initial_shipments=None, 
                 stock_capacity=0, supply_capacity=0, processing_capacity=0, processing_ratio=0,
                 last_level=False, stock_cost=0, supply_cost=0, processing_cost=0, 
                 exceeded_capacity_cost=1, unmet_demand_cost=1, max_leadtime=4, build_info=False):
        self.build_info = build_info
        self.label = label
        self.supply_action = None
        self.ship_action = None
        self.num_actions = 0
        if supply_capacity > 0:
            self.supply_action = SC_Action('SUPPLY', capacity=supply_capacity, costs=supply_cost)
            self.num_actions += 1
            self.max_ship = supply_capacity
        else:
            # Indica o máximo de material que pode chegar por transporte em cada período
            # (será a soma das capacidades das origens)
            # Ver issue #1 para maiores detalhes.
            self.max_ship = 0
        self.processing_capacity = processing_capacity
        self.last_level = last_level
        
        self.initial_stock = initial_stock
        self.initial_supply = initial_supply
        self.initial_shipments = initial_shipments
        
        self.stock = self.initial_stock
        self.stock_capacity = stock_capacity
        self.stock_cost = stock_cost
        self.exceeded_capacity_cost = exceeded_capacity_cost
        self.unmet_demand_cost = unmet_demand_cost
        
        self.processing_ratio = processing_ratio
        self.processing_cost = processing_cost
        
        self.dests = None
        self.shipments = []
        self.max_leadtime = max_leadtime

    def define_destinations(self, dests, costs):
        self.dests = dests
        self.ship_action = SC_Action('SHIP', costs=costs)
        self.num_actions += len(self.dests)
        # Atualiza a capacidade de transporte do destino
        for dest in self.dests:
            dest.max_ship += self.stock_capacity

    def act(self, action_values, leadtimes, time_step, customer_demand=None):
        total_cost = 0
        next_leadtime_idx = 0

        arrived_material = 0
        # O primeiro passo é receber o material que está pra chegar
        while self.shipments and self.shipments[0][0] == time_step:
            _, amount = heapq.heappop(self.shipments)
            arrived_material += amount
            
        # Se é uma fábrica, é necessário processar a matéria-prima, gerando produto
        if self.processing_ratio > 0:
            # A matéria-prima que chegou é somada com matéria-prima anterior que
            # pode ter sobrado como resto do processamento
            material = arrived_material

            # Se chegou mais matéria-prima na fábrica do que ela é capaz de processar,
            # um custo de penalização é calculado e o material excedente é perdido.
            if material > self.processing_capacity:
                total_cost += self.exceeded_capacity_cost*(material - self.processing_capacity)
                if self.build_info:
                    self.est_costs['processing_penalty'] = self.exceeded_capacity_cost*(material - self.processing_capacity)
                    self.est_units['processing_penalty'] = material - self.processing_capacity
                material = self.processing_capacity
            else:
                if self.build_info:
                    self.est_costs['processing_penalty'] = 0
                    self.est_units['processing_penalty'] = 0

            # Calcula a quantidade de produto gerado
            amount = material / self.processing_ratio
            # Calcula o novo resto de matéria-prima do processamento
            total_cost += material * self.processing_cost
            if self.build_info:
                self.est_costs['processing'] = material * self.processing_cost
                self.est_units['processing'] = material
            self.stock += amount
        else:
            self.stock += arrived_material
        
        # Se a quantidade de material que tinha no estoque mais o que chegou for maior que a
        # capacidade, um custo de penalização é gerado e o material excedente é perdido.
        if self.stock > self.stock_capacity:
            total_cost += self.exceeded_capacity_cost*(self.stock - self.stock_capacity)
            if self.build_info:
                self.est_costs['stock_penalty'] = self.exceeded_capacity_cost*(self.stock - self.stock_capacity)
                self.est_units['stock_penalty'] = self.stock - self.stock_capacity
            self.stock = self.stock_capacity
        else:
            if self.build_info:
                self.est_costs['stock_penalty'] = 0
                self.est_units['stock_penalty'] = 0
                
        # Se o nó é um fornecedor, executa a ação de fornecimento
        next_action_idx = 0
        if self.supply_action:
            # A aplicação da ação retorna a quantidade de material a ser fornecido e o custo da operação
            amount, cost = self.supply_action.apply(action_values[next_action_idx])
            next_action_idx += 1
            # adiciona o material para ser fornecido
            if amount > 0:
                self._ship_material(time_step+leadtimes[next_leadtime_idx], amount)
                next_leadtime_idx += 1
            # Contabiliza os custos e estatísticas
            total_cost += cost
            if self.build_info:
                self.est_costs['supply'] = cost
                self.est_units['supply'] = amount            

        # Executa a ação de envio de material para o próximo estágio da cadeia
        if self.ship_action:
            # A aplicação da ação retorna a quantidade de material a ser enviado para cada destino 
            # e o custo de cada da operação
            amounts, costs = self.ship_action.apply(action_values[next_action_idx:], max=self.stock)
            self.stock -= sum(amounts)
            # Trata o envio dos materiais
            for i in range(len(self.dests)):
                if amounts[i] > 0:
                    self.dests[i]._ship_material(time_step+leadtimes[next_leadtime_idx], amounts[i])
                    next_leadtime_idx += 1
            # Contabiliza os custos e estatísticas
            total_cost += sum(costs)
            if self.build_info:
                self.est_costs['ship'] = sum(costs)
                self.est_units['ship'] = sum(amounts)
            
        # Sé um revendedor (nó de último nível) atende a demanda do cliente (o que for possível)
        elif self.last_level:
            max_possible = min(self.stock, customer_demand)
            self.stock -= max_possible
            # Contabiliza os custos e estatísticas
            total_cost += self.unmet_demand_cost * (customer_demand - max_possible)
            if self.build_info:
                self.est_costs['unmet_dem'] = self.unmet_demand_cost * (customer_demand - max_possible)
                self.est_units['unmet_dem'] = customer_demand - max_possible
            

        # Contabiliza os custos e estatísticas do material que ficou em estoque
        total_cost += self.stock*self.stock_cost
        if self.build_info:
            self.est_costs['stock'] = self.stock*self.stock_cost
            self.est_units['stock'] = self.stock

        return total_cost

    def _ship_material(self, time, amount):
        heapq.heappush(self.shipments, (time,amount))

    def reset(self):
        self.stock = self.initial_stock
        self.shipments.clear()
        if self.initial_supply:
            for i in range(len(self.initial_supply)):
                self._ship_material(i+1, self.initial_supply[i])
        if self.initial_shipments:
            for i in range(len(self.initial_shipments)):
                self._ship_material(i+1, self.initial_shipments[i])
        
        # Atributos estatísticos para depuração
        if self.build_info:
            self.est_costs = {'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'processing_penalty':0, 'ship':0, 'unmet_dem':0}
            self.est_units = {'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'processing_penalty':0, 'ship':0, 'unmet_dem':0}

    def num_expected_actions(self):
        return self.num_actions

    def render(self):
        print(self.__str__(), end='')

    def is_last_level(self):
        return self.last_level

    def build_observation(self, shipments_range):
        """ Observação é [estoque, ship1, ship2, ...]
            Onde ship1 é a soma dos carregamentos que chegam no próximo período,    
            ship2 no período seguinte, e assim por diante, até que o último valor é a soma
            de todos os carregamentos daquele período em diante.

            Os valores são normalizados, ou seja, o valor de estoque é a porcentagem
            da capacidade do estoque.
            Os valores dos carregamentos, por enquanto, serão dados também como porcentagem
            da capacidade do estoque, assumindo como esse o limite superior para transporte.
        """
        # a primeira informação é o estoque atual.
        # - No caso das fábricas inclui a fração de matéria-prima pendente (se existir)
        current_stock = self.stock

        obs = [current_stock/self.stock_capacity]

        # Se não tem nenhum carregamento pra chegar
        # Cria os carregamentos vazios
        if not self.shipments:
            obs += [0]*(shipments_range[1]-shipments_range[0]+1)
            return obs
        else:
            # os carregamentos são dados pelo total de material em transporte em cada período.
            # exceto o último, que é dado pelo material em transporte daquele período em diante

            # Vamos primeiro tratar os períodos antes do último do range
            ship_idx = 0
            for time_step in range(shipments_range[0], shipments_range[1]):
                obs.append(0)
                while ship_idx < len(self.shipments) and self.shipments[ship_idx][0] == time_step:
                    obs[-1] += self.shipments[ship_idx][1]
                    ship_idx += 1
                # normaliza a quantidade de material em transporte
                obs[-1] /= self.max_ship
            
            # Agora vamos tratar o último período do range
            time_step = shipments_range[1]
            obs.append(0)
            while ship_idx < len(self.shipments):
                obs[-1] += self.shipments[ship_idx][1]
                ship_idx += 1
            # normaliza a quantidade de material em transporte
            obs[-1] /= self.max_ship*(self.max_leadtime-(shipments_range[1]-shipments_range[0]))

            return obs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        desc = self.label + ' ['
        for i in range(len(self.shipments)):
            desc += str(self.shipments[i][0]) + ' ' + str(np.round(self.shipments[i][1],1)) + ', '
        desc += '] [' + str(np.round(self.stock,1)) + '] '
        return desc

class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}
    def __init__(self, nodes_info, unmet_demand_cost=1000, exceeded_capacity_cost=1000,
                 demand_range=(10,20), demand_std=None, demand_sen_peaks=None, avg_demand_range=None, 
                 processing_ratio=3, stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2,
                 total_time_steps=360, seed=None,
                 build_info=False, demand_perturb_norm=False):
        """
            :param stochastic_leadtimes: (bool) indica se os leadtimes serão constantes ou estocásticos.
            :param avg_leadtime: (int) valor constante do leadtime ou a média dos leadtimes se eles forem estocásticos.
                                 Valores seguem distribuição de Poisson truncada, ou seja, evitando valor zero.
            :param max_leadtime: (int) no caso de leadtimes estocásticos indica o limite de leatime 
                                 (valores maiores que o limite são igualados ao limite). Cuidado: isso afeta a média real.
                                 No caso de leadtime constante, só faz sentido que seja igual ao leadtime constante.
        """
        
        def create_nodes(nodes_info):
            nodes_dict = {}
            self.nodes = []
            self.last_level_nodes = []
            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                
                processing_cost = node_info.get('processing_cost', 0)
                if processing_cost > 0:
                    node_processing_ratio = processing_ratio
                else:
                    node_processing_ratio = 0
                    
                node = SC_Node(node_name,
                               initial_stock=node_info.get('initial_stock', 0),
                               initial_supply=node_info.get('initial_supply', None),
                               initial_shipments=node_info.get('initial_shipments', None),
                               stock_capacity=node_info.get('stock_capacity', float('inf')),
                               supply_capacity=node_info.get('supply_capacity', 0),
                               processing_capacity=node_info.get('processing_capacity', 0),                               
                               processing_ratio=node_processing_ratio,
                               last_level=node_info.get('last_level', False),
                               stock_cost=node_info.get('stock_cost', 0),
                               supply_cost=node_info.get('supply_cost', 0),
                               processing_cost=processing_cost,
                               exceeded_capacity_cost=exceeded_capacity_cost,
                               unmet_demand_cost=unmet_demand_cost,
                               max_leadtime=self.max_leadtime,
                               build_info=self.build_info)                
                nodes_dict[node_name] = node
                self.nodes.append(node)
                if node.is_last_level():
                    self.last_level_nodes.append(node)

            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                node = nodes_dict[node_name]
                if 'destinations' in node_info:
                    node.define_destinations([nodes_dict[dest_name] for dest_name in node_info['destinations']],
                                             node_info['dest_costs'])

        self.build_info   = build_info
        self.stochastic_leadtimes = stochastic_leadtimes
        self.avg_leadtime = avg_leadtime
        self.max_leadtime = max_leadtime
                    
        create_nodes(nodes_info)
        self.total_time_steps = total_time_steps                
        self.rand_generator = np.random.RandomState(seed)
        self.demand_range = demand_range
        self.demand_range_value = demand_range[1]-self.demand_range[0]
        self.demand_std = demand_std
        self.demand_sen_peaks = demand_sen_peaks
        if avg_demand_range:
            self.minavg_demand = avg_demand_range[0]
            self.maxavg_demand = avg_demand_range[1]
        else:
            self.minavg_demand = None
            self.maxavg_demand = None
        self.demand_perturb_norm = demand_perturb_norm
        
        if stochastic_leadtimes:
            self.count_leadtimes_per_timestep = 0
            for node in self.nodes:            
                if node.supply_action:
                    self.count_leadtimes_per_timestep += 1
                self.count_leadtimes_per_timestep += len(node.dests) if node.dests else 0

        # Não suporta demanda fixa (apenas para evitar ficar fazendo if toda hora 
        # para testar isso na hora de montar o estado)
        assert self.demand_range[0] != self.demand_range[1]

        # Definição dos espaços de ações e de estados
        action_space_size = 0
        for node in self.nodes:
            action_space_size += node.num_expected_actions()            

        # As observações (estados) são dados pela:
        # - len(self.last_level_nodes)      : Demandas, sendo 1 para cada varejista
        # - len(self.nodes)*1               : Níveis de estoque, 1 para cada nó da cadeia.
        # - len(self.nodes)*(avg_leadtime)  : Materiais em transporte; sendo l o leadtime médio, considera, para cada nó
        #                                     a quantidade de material em transporte no período t+1, t+2, ... t+(l-1), sum_{i>=l}t+i
        # - 1                               : número de períodos para terminar o episódio
        obs_space_size = len(self.last_level_nodes)+len(self.nodes)*(1+avg_leadtime)+1        

        # O action_space é tratado como de [0,1] no código, então quando a ação é recebida o valor
        # é desnormalizado
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(action_space_size,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_space_size,))        

        self.current_state = None

    def reset(self):
        for node in self.nodes:
            node.reset()
        self.time_step = 0

        self.current_reward = 0
        self.episode_rewards = 0
        # gerando as demandas de todo o episódio
        self.customer_demands = generate_demand(self.rand_generator, (self.total_time_steps+1, len(self.last_level_nodes)), 
                                                self.total_time_steps, self.demand_range[0], self.demand_range[1],
                                                std=self.demand_std, sen_peaks=self.demand_sen_peaks,
                                                minavg=self.minavg_demand, maxavg=self.maxavg_demand,
                                                perturb_norm=self.demand_perturb_norm)
        
        if self.stochastic_leadtimes:
            # Gerando os leadtimes para todo o episódio
            # - O formato é: time_steps x número de leadtimes por período (fornecedores e destinos de cada nó)
            # - Para evitar valores iguais a zero, a distribuição de Poisson é gerada com média `avg_leadtime-1` e os valores
            #   são todos somados com 1.
            # - E para evitar valores maiores que o máximo esperado, os valores são cortados com `np.clip`.
            self.leadtimes = 1 + self.rand_generator.poisson(lam=self.avg_leadtime-1, 
                                        size=(self.total_time_steps, self.count_leadtimes_per_timestep))
            self.leadtimes = np.clip(self.leadtimes, 1, self.max_leadtime)

        self.current_state = self._build_observation()

        # Estatísticas do episódio
        if self.build_info:
            self.est_episode = self._initial_est_episode()
        else:
            self.current_info = {}

        return self.current_state
    
    def _initial_est_episode(self):
        return {'rewards':0, 
                'costs':{
                    'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'processing_penalty':0, 'ship':0, 'unmet_dem':0},
                'units':{
                    'stock':0, 'stock_penalty':0, 'supply':0, 'processing':0, 'processing_penalty':0, 'ship':0, 'unmet_dem':0}
                }
        
    def _denormalize_action(self, action):
        return (action+1)/2
        
    def _normalize_obs(self, obs):
        return np.clip(obs*2 - 1, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        action = self._denormalize_action(action)
        
        self.time_step += 1        

        total_cost = 0

        next_action_idx = 0
        next_customer = 0
        for node in self.nodes:
            actions_to_apply   = action[next_action_idx:next_action_idx+node.num_expected_actions()]
            if self.stochastic_leadtimes:
                leadtimes_to_apply = self.leadtimes[self.time_step-1, next_action_idx:next_action_idx+node.num_expected_actions()]
            else:
                leadtimes_to_apply = node.num_expected_actions()*[self.avg_leadtime]
            
            next_action_idx   += node.num_expected_actions()
            
            if node.last_level:
                demand = self.customer_demands[self.time_step-1, next_customer]
                next_customer += 1
            else:
                demand = None
            
            cost = node.act(actions_to_apply, leadtimes_to_apply, self.time_step, customer_demand=demand)
            total_cost += cost

        self.current_reward = -total_cost
        self.episode_rewards += self.current_reward
        self.current_state = self._build_observation()

        is_terminal = self.time_step == self.total_time_steps

        if self.build_info:
            self._update_statistics()     
            self.current_info = self._build_return_info()
        
        #print(self.customer_demands[self.time_step,:], end=' ')

        return (self.current_state, self.current_reward, is_terminal, self.current_info)

    def _update_statistics(self):
        self.est_episode['rewards'] += self.current_reward
        costs = self.est_episode['costs']
        units = self.est_episode['units']
        for node in self.nodes:
            for key,value in node.est_costs.items():
                costs[key] += value
            for key,value in node.est_units.items():
                units[key] += value

    def _build_observation(self):
        """ Uma observação será formada:
            - Pelas demandas dos clientes
            - Pela obervação de cada nó. Que tem:
                - A quantidade de material em estoque.
                - O quanto de material está chegando nos períodos seguintes.
        """ 
        # Primeiro guardamos as demandas (normalizadas)
        demands_obs = (self.customer_demands[self.time_step,:] - self.demand_range[0])/(self.demand_range_value)
            
        # Depois pegamos os dados de cada nó
        nodes_obs = []
        for node in self.nodes:
            nodes_obs += node.build_observation((self.time_step+1, self.time_step+self.avg_leadtime))        

        # A observação é concatenação das demandas, com os dados nos nós mais quantos períodos 
        # faltam para terminar o episódio (normalizado)
        obs = np.concatenate((demands_obs, nodes_obs, [(self.total_time_steps-self.time_step)/self.total_time_steps]))
        
        # Por fim, normalizamos o estado para a faixa [-1,1]
        norm_obs = self._normalize_obs(obs)

        return norm_obs

    def _build_return_info(self):
        # O prefixo 'sc_' serve pra identificar as chaves que se refere ao ambiente
        # porque a biblioteca Stable Baselines inclui informação adicional
        return {'sc_episode':self.est_episode}

    def render(self, mode='human'):
        print('TIMESTEP:', self.time_step)
        for node in self.nodes:
            node.render()
            print()                 
        print('Next demands  :', self.customer_demands[self.time_step,:])
        print('Current state :', self.current_state)
        print('Current reward:', round(self.current_reward,3))
        print('='*30)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)

if __name__ == '__main__':
    demand_range     = (10,20)
    stock_capacity   = 300
    supply_capacity  = 50
    processing_capacity = 50
    processing_ratio = 3
    stochastic_leadtimes = True
    avg_leadtime= 2
    max_leadtime= 4
    stock_cost  = 1
    dest_cost   = 2*stock_cost
    supply_cost = 5*stock_cost
    processing_cost   = 2*supply_cost
    # Quanto custa para produzir e entregar uma unidade de produto (sem usar estoque)
    # Obs: na verdade custo de transporte seria 2*avg_leadtime*cost, porque não paga transporte no fornecimento
    product_cost = supply_cost + 3*avg_leadtime*dest_cost + processing_cost 
    # O custo de demanda não atendida é duas vezes o custo de produzir (como se comprasse do concorrente).
    unmet_demand_cost = 2*product_cost
    # O custo de excesso de estoque talvez pudesse nem existir, já que o custo já incorrido no material
    # é perdido. Mas podemos considerar também que existiria um custo de desfazer do material.
    exceeded_capacity_cost = 10*stock_cost
    
    total_time_steps = 5
    
    nodes_info = {}
    nodes_info['Supplier 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Supplier 2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  1'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Factory  2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Wholesal 2'] = {'initial_stock':15, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[dest_cost]*2}
    nodes_info['Retailer 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}
    nodes_info['Retailer 2'] = {'initial_stock':20, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}

    env = SupplyChainEnv(nodes_info, demand_range=demand_range, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_capacity_cost=exceeded_capacity_cost, processing_ratio=processing_ratio, 
                         stochastic_leadtimes=stochastic_leadtimes, avg_leadtime=avg_leadtime, max_leadtime=max_leadtime,
                         total_time_steps=total_time_steps)
    env.action_space.seed(0)
    env.seed(0)
    env.reset()
    env.render()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

