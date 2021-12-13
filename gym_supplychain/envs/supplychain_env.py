import numpy as np
import heapq

import gym
from gym import spaces

from .demands_generator import generate_demand

# TODO: definir valores padrões ao criar o ambiente (ideal é que mantivesse o comportamento anterior)

# TODO: melhorar desempenho: usar sempre numpy array ao invés de listas.
# TODO: melhorar desempenho: usar Counter para atualiza o dicionário de estatísticas


class SC_Action:
    """ Define uma tipo de ação do ambiente da Cadeia de Suprimentos, para um tipo de produto.

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

    def apply(self, action_values, maximum=None):
        """ Aplica o valor de ação recebido, podendo ter um limite variável.

            :param action_value: (float or list) valor (ou lista de valores) entre 0 e 1 (porcentagem) referente à ação a ser aplicada.
            :param max: (int) valor máximo da ação para essa decisão específica.
            :return: (int or list) retorna o(s) valor(es) inteiro(s) correspondente(s) à ação.
        """
        if self.action_type == 'SUPPLY':
            # No caso de fornecimento basta usar o valor percentual no máximo de
            # valor inteiro possível (mínimo entre capacidade e máximo recebido)
            if maximum is None:
                limit = self.capacity
            else:
                limit = min(self.capacity, maximum)
            supplied_amount = action_values*limit
            return supplied_amount, supplied_amount*self.costs
        elif self.action_type == 'SHIP':
            
            # Definindo a quantidade máxima de material disponível para envio
            if self.capacity is None:
                limit = maximum
            else:
                limit = min(self.capacity, maximum)
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
    
    def calculate_costs(self, amounts):
        return [amounts[i]*self.costs[i] for i in range(len(amounts))]

class SC_Node:
    """ Define um nó da Cadeia de Suprimentos        

        Cada nó tem:
        - Um identificador.
        - Uma capacidade de estoque para cada produto.
        - Uma capacidade de fornecimento para cada produto.
        - Uma indicação se é nó do último nível (estágio).
        - Uma quantidade de material em estoque para cada produto.
        - Uma lista de ações possíveis.
        - O número total de valores de ação esperados.
        - Uma lista de destinos.
        - Uma fila de material para chegar de cada produto.
        @param shipments_group_size: os envios de material mais antigos podem ser agrupados em um único envio. 
                                     Parâmetro deve estar entre 1 e self.max_leadtime.
    """
    def __init__(self, label, num_products=1, initial_stock=0, initial_supply=None, initial_shipments=None, 
                 stock_capacity=0, supply_capacity=0, processing_capacity=0, processing_ratio=0,
                 last_level=False, stock_cost=0, supply_cost=0, processing_cost=0, 
                 exceeded_stock_capacity_cost=1, exceeded_process_capacity_cost=1, exceeded_ship_capacity_cost=1,
                 unmet_demand_cost=1, avg_leadtime=2, max_leadtime=4, build_info=False, shipments_group_size=1):
        # ALTERAÇÃO: Adicionado parâmetro de leadtime médio

        self.build_info = build_info
        self.label = label
        self.num_products = num_products
        self.supply_actions = [None]*self.num_products
        self.ship_actions = [None]*self.num_products
        self.num_supply_actions = 0
        self.num_ship_actions = 0
        
        supply_capacity = self._treat_int_or_list_param(supply_capacity)
        supply_cost = self._treat_int_or_list_param(supply_cost)

        # se o nó é capaz de fornecer algum produto
        if max(supply_capacity) > 0:
            # cria uma ação de fornecimento para cada produto
            for prod, capacity in enumerate(supply_capacity):
                if capacity > 0:
                    self.supply_actions[prod] = SC_Action('SUPPLY', capacity=capacity, costs=supply_cost[prod])
                    self.num_supply_actions += 1
                else:
                    self.supply_actions[prod] = None
            # Max_ship é o valor usado na normalização do estado.
            # No caso dos fornecedores ele depende do produto e é a capacidade de fornecimento.
            self.max_ship = supply_capacity.copy()
        else:
            # Max_ship é o valor usado na normalização do estado.
            # No caso dos outros nós será a própria capacidade transporte. Ela não depende do produo
            # mas vou replicar o valor por produto apenas para simplificar o código da normalização do estado
            self.max_ship = [0]*self.num_products

        # A capacidade de processamento é a total e não por produto
        self.processing_capacity = processing_capacity
        
        self.last_level = last_level

        self.initial_stock = self._treat_int_or_list_param(initial_stock)
        self.initial_supply = initial_supply
        self.initial_shipments = initial_shipments
        
        self.stock = self.initial_stock
        self.stock_capacities = self._treat_int_or_list_param(stock_capacity)
        self.stock_cost = self._treat_int_or_list_param(stock_cost)
        self.exceeded_stock_capacity_cost = exceeded_stock_capacity_cost
        self.exceeded_process_capacity_cost = exceeded_process_capacity_cost
        self.exceeded_ship_capacity_cost = exceeded_ship_capacity_cost
        self.unmet_demand_cost = unmet_demand_cost
        
        self.processing_ratio = self._treat_int_or_list_param(processing_ratio)
        self.processing_cost = self._treat_int_or_list_param(processing_cost)
        
        self.dests = None
        self.shipments_by_prod = [[] for i in range(self.num_products)]
        self.avg_leadtime = avg_leadtime
        self.max_leadtime = max_leadtime
        # ALTERAÇÃO: acrescentado parâmetro shipments_group_size
        self.shipments_group_size = shipments_group_size


    def _treat_int_or_list_param(self, param, default_value=0):
        # se foi passada uma lista, ou ela deve ser vazia ou ter um valor por produto
        if type(param) is list:
            if len(param) > 0:
                assert len(param) == self.num_products
            else:
                param = [default_value]*self.num_products
        # se não foi passada uma lista, e sim um valor único ele será replicado para todos os produtos
        elif type(param) is int:
            param = [param]*self.num_products
        else:
            raise ValueError(f"Invalid param: '{param}' should be an int or a list with one value per product")
        
        return param

    def define_destinations(self, dests, ship_capacities_by_dest, ship_costs):
        self.dests = dests
        self.ship_capacities = ship_capacities_by_dest
        for prod, stock_capacity in enumerate(self.stock_capacities):
            if stock_capacity > 0:
                self.ship_actions[prod] = SC_Action('SHIP', capacity=stock_capacity, costs=ship_costs[prod])
                self.num_ship_actions += len(self.dests)
            else:
                self.ship_actions[prod] = None

        # Atualiza a capacidade de transporte do destino
        for dest_idx, dest in enumerate(self.dests):
            for prod in range(self.num_products):
                dest.max_ship[prod] += ship_capacities_by_dest[dest_idx]

    def act(self, action_values, leadtimes, time_step, customer_demand=None):
        total_cost = 0
        next_leadtime_idx = 0
        
        # Zerando custos e unidades anteriores
        # TODO: #20 No método `act` incializar custos copiando estrutura zerada
        if self.build_info:
            for key in self.est_costs:
                self.est_costs[key] = [0]*self.num_products
            for key in self.est_units:
                self.est_units[key] = [0]*self.num_products

        arrived_material = np.zeros(self.num_products)
        # O primeiro passo é receber o material que está pra chegar
        for prod, shipments in enumerate(self.shipments_by_prod):
            # ALTERAÇÃO: a informação de tempo de chegada agora é a segunda da tupla
            while shipments and shipments[0][1] == time_step:
                _, _, amount = heapq.heappop(shipments)
                arrived_material[prod] += amount
        
        # Adiciona o material no estoque
        self.stock += arrived_material
        
        # Se a quantidade de material que tinha no estoque mais o que chegou for maior que a
        # capacidade, um custo de penalização é gerado e o material excedente é perdido.
        for prod in range(self.num_products):
            if self.stock[prod] > self.stock_capacities[prod]:
                # Calculando penalização
                total_cost += self.exceeded_stock_capacity_cost*(self.stock[prod] - self.stock_capacities[prod])
                if self.build_info:
                    self.est_costs['stock_pen'][prod] = self.exceeded_stock_capacity_cost*(self.stock[prod] - self.stock_capacities[prod])
                    self.est_units['stock_pen'][prod] = self.stock[prod] - self.stock_capacities[prod]
                # Descartando material excedente
                self.stock[prod] = self.stock_capacities[prod]
                
        # Se o nó é um fornecedor, executa as ações de fornecimento
        next_action_idx = 0
        if self.num_supply_actions > 0:
            for prod, supply_action in enumerate(self.supply_actions):
                # Se tem ação de fornecimento para o produto
                if supply_action:
                    # A aplicação da ação retorna a quantidade de material a ser fornecido e o custo da operação
                    amount, cost = supply_action.apply(action_values[next_action_idx])
                    next_action_idx += 1
                    # adiciona o material para ser fornecido
                    if amount > 0:
                        # ALTERAÇÃO: acrescentado parâmetro do momento de envio
                        self._ship_material(time_step, time_step+leadtimes[next_leadtime_idx], prod, amount)
                        next_leadtime_idx += 1
                    # Contabiliza os custos e estatísticas
                    total_cost += cost
                    if self.build_info:
                        self.est_costs['supply'][prod] = cost
                        self.est_units['supply'][prod] = amount

        # Se não é um revendedor
        if not self.last_level:
            # A princípio toda a capacidade de transporte para cada destino está disponível.
            # À medida que os materiais forem sendo enviados, a quantidade é descontada da capacidade disponível
            available_ship_capacities = self.ship_capacities.copy()        
            # No caso de uma fábrica, a princípio toda a capacidade de processamento está disponível.
            # E o mesmo tratamento é feito
            available_processing_capacity = self.processing_capacity        

            next_prod_leadtime_idx = next_leadtime_idx
            # Executa as ações de envio de material para o próximo estágio da cadeia, para cada produto
            for prod, ship_action in enumerate(self.ship_actions):
                # Se tem ação de envio para esse produto
                if ship_action:

                    exceeded_ship_capacity = 0
                    exceeded_processing_capacity = 0

                    # O material a ser considerado para envio é todo o estoque atual do produto em questão
                    available_material = self.stock[prod]
                    
                    # Se tem algum material a ser enviado
                    if available_material > 0:
                    
                        # A aplicação da ação retorna a quantidade de material a ser enviado para cada destino 
                        # e o custo de cada da operação.
                        # Obs: as ações se referem à porcentagem da quantidade de material atualmente em estoque.
                        amounts, _ = ship_action.apply(action_values[next_action_idx:next_action_idx+len(self.dests)], maximum=available_material)                        
                        
                        # As quantidades a serem enviadas geralmente são as mesmas que saíram do estoque
                        # isso pode ser diferente para as fábricas, porque deve-se aplicar as razões de processamento.
                        amounts_to_ship = amounts.copy()
                        
                        # Antes de enviar o material é necessário verificar sua viabilidade, ou seja,
                        # se atende às capacidades de transporte e processamento,
                        # além de aplicar a razão de processamnento
                                            
                        if self.processing_capacity > 0:
                            for i, amount in enumerate(amounts):
                                if amount > 0:
                                    # se o material a ser enviado para um destino excede a capacidade de processamento ainda disponível
                                    if amount > available_processing_capacity:
                                        # contabiliza o que excede a capacidade da fábrica
                                        exceeded_processing_capacity += amount - available_processing_capacity
                                        # desconsidera o material excedente
                                        amounts[i] = available_processing_capacity
                                    # atualiza a capacidade ainda disponível da fábrica
                                    available_processing_capacity -= amounts[i]
                                # trata a razão de processamento no material a ser enviado
                                amounts_to_ship[i] = amounts[i]/self.processing_ratio[prod]
                        
                        for i, amount in enumerate(amounts_to_ship):
                            if amount > 0:
                                # se o material a ser enviado para um destino excede a capacidade de transporte ainda disponível
                                # para o destino
                                if amount > available_ship_capacities[i]:
                                    # contabiliza o que excede a capacidade de transporte
                                    exceeded_ship_capacity += amount - available_ship_capacities[i]
                                    # desconsidera o material excedente
                                    amounts_to_ship[i] = available_ship_capacities[i]
                                    # já que a quantidade de material a ser enviada foi alterada, precisamos alterar também
                                    # a quantidade de material que está saindo dos estoques.
                                    if self.processing_capacity > 0:
                                        amounts[i] = amounts_to_ship[i]*self.processing_ratio[prod]
                                    else:
                                        amounts[i] = amounts_to_ship[i]
                                    # atualiza a capacidade ainda disponível de transporte
                                    available_ship_capacities[i] -= amounts[i]

                        # Retira do estoque o material que está saindo
                        total_leaving_stock = sum(amounts)
                        self.stock[prod] -= total_leaving_stock

                        # Se é uma fábrica, precisamos contabilizar o custo de processamento
                        # (que se refere à matéria-prima) e o custo de transporte será referente ao
                        # produto final que foi enviado.
                        if self.processing_capacity > 0:                        
                            total_cost += total_leaving_stock * self.processing_cost[prod]
                            if self.build_info:
                                self.est_costs['process'][prod] = total_leaving_stock * self.processing_cost[prod]
                                self.est_units['process'][prod] = total_leaving_stock                        
                        
                        # Trata o envio dos materiais
                        for i in range(len(self.dests)):
                            # Se tem algum material a ser enviado
                            if amounts_to_ship[i] > 0:
                                # ALTERAÇÃO: acrescentado parâmetro do momento de envio
                                self.dests[i]._ship_material(time_step, time_step+leadtimes[next_leadtime_idx], prod, amounts_to_ship[i])
                            next_leadtime_idx += 1

                        # Contabiliza os custos e estatísticas de transporte
                        # Os custos são atualizados porque excessos podem ter sido descartados
                        ship_costs = sum(ship_action.calculate_costs(amounts_to_ship))
                        total_cost += ship_costs
                        if self.build_info:
                            self.est_costs['ship'][prod] = ship_costs
                            self.est_units['ship'][prod] = sum(amounts_to_ship)
            
                    # Agora que o processamento e transporte foi tratado, vamos contabilizar os possíveis
                    # custos de penalização por ter excedido as capacidades de processamento e transporte
                    
                    total_cost += self.exceeded_process_capacity_cost * exceeded_processing_capacity
                    if self.build_info:
                        self.est_costs['process_pen'][prod] = self.exceeded_process_capacity_cost*exceeded_processing_capacity
                        self.est_units['process_pen'][prod] = exceeded_processing_capacity
                    
                    total_cost += self.exceeded_ship_capacity_cost * exceeded_ship_capacity
                    if self.build_info:
                        self.est_costs['ship_pen'][prod] = self.exceeded_ship_capacity_cost*exceeded_ship_capacity
                        self.est_units['ship_pen'][prod] = exceeded_ship_capacity
                    
                    # Apontando para as ações do próximo produto
                    next_action_idx += len(self.dests)

                    # Volta a posição dos lead times de transporte porque eles são os mesmos independente do produto.
                    next_leadtime_idx = next_prod_leadtime_idx                                    


        # Se é um revendedor (nó de último nível) atende a demanda do cliente por cada produto (o que for possível)
        else:
            for prod in range(self.num_products):
                max_possible = min(self.stock[prod], customer_demand[prod])
                self.stock[prod] -= max_possible
                # Contabiliza os custos e estatísticas
                total_cost += self.unmet_demand_cost * (customer_demand[prod] - max_possible)
                if self.build_info:
                    self.est_costs['unmet_dem'][prod] = self.unmet_demand_cost * (customer_demand[prod] - max_possible)
                    self.est_units['unmet_dem'][prod] = customer_demand[prod] - max_possible

        # Contabiliza os custos e estatísticas do material que ficou no estoque de cada produto
        for prod in range(self.num_products):
            total_cost += self.stock[prod]*self.stock_cost[prod]
            if self.build_info:
                self.est_costs['stock'][prod] = self.stock[prod]*self.stock_cost[prod]
                self.est_units['stock'][prod] = self.stock[prod]

        return total_cost

    def _ship_material(self, dispatch_time, arrive_time, product, amount):
        # ALTERAÇÃO: Adicionado o parâmetro dispatch_time (que representa o momento de envio do material).
        # Obs.: o valor pode ser negativo.

        # TODO: #21 Melhorar desempenho do material em transporte

        # ALTERAÇÃO: agora o heap é ordenado pelo momento que o material foi despachado
        heapq.heappush(self.shipments_by_prod[product], (dispatch_time, arrive_time, amount))

    def reset(self):
        self.stock = self.initial_stock
        self.shipments_by_prod = [[] for i in range(self.num_products)]
        # ALTERAÇÃO: Acrescentado parâmetro de momento de envio na chamada do método _ship_material (considera
        #            que material foi enviado e o lead time foi o médio)
        if self.initial_supply:
            for prod in range(self.num_products):
                for i in range(len(self.initial_supply[prod])):                
                    self._ship_material(i+1-self.avg_leadtime, i+1, prod, self.initial_supply[prod][i])
        if self.initial_shipments:
            for prod in range(self.num_products):
                for i in range(len(self.initial_shipments[prod])):                
                    self._ship_material(i+1-self.avg_leadtime, i+1, prod, self.initial_shipments[prod][i])
        
        # Atributos estatísticos para depuração
        if self.build_info:
            self.est_costs = {'stock':[], 'stock_pen':[], 'supply':[], 'process':[], 'process_pen':[], 'ship':[], 'ship_pen':[], 'unmet_dem':[]}
            self.est_units = {'stock':[], 'stock_pen':[], 'supply':[], 'process':[], 'process_pen':[], 'ship':[], 'ship_pen':[], 'unmet_dem':[]}

    def num_expected_actions(self):
        return self.num_supply_actions+self.num_ship_actions

    def render(self):
        print(self.__str__(), end='')

    def is_last_level(self):
        return self.last_level

    def build_observation(self, time_step):
        """ 
        Cria uma observação do nó.
        @param time_step: período atual    
        """
        # ALTERAÇÃO: retirado parâmetro shipments_range e acrescentado time_step (agora o material em transporte 
        #            aparece no estado de acordo com o momento de envio)
        
        # a primeira informação é o estoque atual.
        current_stock = self.stock

        obs = [current_stock[i]/self.stock_capacities[i] for i in range(len(current_stock))]

        # Depois as informações de transporte são agrupadas por produto
        for prod, shipments in enumerate(self.shipments_by_prod):
            # Se não tem nenhum carregamento pra chegar, cria os carregamentos vazios
            if not shipments:
                obs += [0]*(self.max_leadtime)
            else:
                # os carregamentos são dados pelo total de material em transporte despachado em cada período.
                # Os carregamentos mais antigos podem ser agrupados de acordo com o atributo shipments_group_size

                # inicializa posição na lista de envios
                ship_idx = 0
                # inicializa período mais antigo de envio
                t = time_step-self.max_leadtime+1
                # tamanho do primeiro grupo é dado pelo atributo shipments_group_size. Os demais valores não são agrupados.
                group_size = self.shipments_group_size
                
                # percorre os possíveis períodos de envio
                while t < time_step+1:
                    # adiciona um campo na observação
                    obs.append(0)
                    # enquanto ainda existem envios a serem tratados e os envios devem fazer parte do campo atual
                    while ship_idx < len(shipments) and shipments[ship_idx][0] in range(t, t+group_size):
                        # adiciona a quantidade de material do envio em questão
                        obs[-1] += shipments[ship_idx][1]
                        # passa para o próximo envio
                        ship_idx += 1
                    # normaliza a quantidade de material em transporte
                    obs[-1] /= self.max_ship[prod]*group_size

                    # o próximo período a ser avaliado será o posterior ao último grupo tratado
                    t += group_size
                    # apenas os envios mais antigos são agrupados, os demais valores não são agregados
                    group_size = 1

        return obs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        desc = f'{self.label} ('
        for shipments in self.shipments_by_prod:
            desc += f'['
            for i in range(len(shipments)):
                desc += f'{shipments[i][0]}-{shipments[i][1]} {np.round(shipments[i][2],1)}, '
            desc += f']'
        desc += f') [{np.round(self.stock,1)}] '
        return desc

class SupplyChainEnv(gym.Env):
    """ OpenAI Gym Environment for Supply Chain Environments
    """
    #metadata = {'render.modes': ['human']}
    def __init__(self, nodes_info, num_products=1, unmet_demand_cost=1000, 
                 exceeded_stock_capacity_cost=1000, exceeded_process_capacity_cost=1000,
                 exceeded_ship_capacity_cost=1000, 
                 demand_config_by_product=False,
                 demand_range=(10,20), demand_std=None, demand_sen_peaks=None, avg_demand_range=None, 
                 processing_ratio=3, stochastic_leadtimes=False, avg_leadtime=2, max_leadtime=2,
                 total_time_steps=360, seed=None,
                 build_info=False, demand_perturb_norm=False, shipments_group_size=1):
        """
            :param stochastic_leadtimes: (bool) indica se os leadtimes serão constantes ou estocásticos.
            :param avg_leadtime: (int) valor constante do leadtime ou a média dos leadtimes se eles forem estocásticos.
                                 Valores seguem distribuição de Poisson truncada, ou seja, evitando valor zero.
            :param max_leadtime: (int) no caso de leadtimes estocásticos indica o limite de leatime 
                                 (valores maiores que o limite são igualados ao limite). Cuidado: isso afeta a média real.
                                 No caso de leadtime constante, só faz sentido que seja igual ao leadtime constante.
            :param total_time_steps: (int) número de passos de tempo que o ambiente simulará.
            :param seed: (int) valor para inicializar o gerador de números aleatórios.
            :param build_info: (bool) indica se a informação de debug deve ser construída durante a simulação.
            :param demand_perturb_norm: (bool) indica se a demanda deve ser perturbada com uma distribuição normal.
            :param shipments_group_size: (int) indica se os materiais em transporte, enviados há mais tempo, devem ser agrupados no estado.
        """
        
        def create_nodes(nodes_info):
            nodes_dict = {}
            self.nodes = []
            self.last_level_nodes = []
            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                
                processing_cost = node_info.get('processing_cost', 0)
                # if type(processing_cost) is int:
                #     if processing_cost == 0:
                #         node_processing_ratio = 0
                #     else:
                #         node_processing_ratio = node_processing_ratio
                # else:
                #     if all(p == 0 for p in processing_cost):
                #         node_processing_ratio = 0
                #     else:
                #         node_processing_ratio = processing_ratio
                
                if ((type(processing_cost) is int and processing_cost == 0) or 
                    (type(processing_cost) is list and sum(processing_cost) == 0)):
                    node_processing_ratio = 0
                else:
                    node_processing_ratio = processing_ratio
                    
                node = SC_Node(node_name,
                               num_products=num_products,
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
                               exceeded_stock_capacity_cost=exceeded_stock_capacity_cost,
                               exceeded_process_capacity_cost=exceeded_process_capacity_cost,
                               exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                               unmet_demand_cost=unmet_demand_cost,
                               avg_leadtime=self.avg_leadtime,
                               max_leadtime=self.max_leadtime,
                               build_info=self.build_info,
                               shipments_group_size=shipments_group_size)                
                nodes_dict[node_name] = node
                self.nodes.append(node)
                if node.is_last_level():
                    self.last_level_nodes.append(node)

            for node_name in nodes_info:
                node_info = nodes_info[node_name]
                node = nodes_dict[node_name]
                if 'destinations' in node_info:
                    node.define_destinations([nodes_dict[dest_name] for dest_name in node_info['destinations']],
                                             node_info['ship_capacity'],
                                             node_info['dest_costs'])

        self.num_products = num_products
        self.build_info   = build_info
        self.stochastic_leadtimes = stochastic_leadtimes
        self.avg_leadtime = avg_leadtime
        self.max_leadtime = max_leadtime
        
        # O agrupamento dos materiais em transporte enviados há mais tempo deve ter tamanho entre 1 (não agrupado) e max_leadtime (um único grupo).
        assert 1 <= shipments_group_size <= self.max_leadtime
                    
        create_nodes(nodes_info)
        self.total_time_steps = total_time_steps                
        self.rand_generator = np.random.RandomState(seed)
        
        self.demand_config_by_product = demand_config_by_product
        
        self.demand_range = demand_range
        self.demand_std = demand_std
        self.demand_sen_peaks = demand_sen_peaks
        self.minavg_demand = None
        self.maxavg_demand = None
        self.demand_perturb_norm = demand_perturb_norm

        if not self.demand_config_by_product:
            self.demand_range_value = self.demand_range[1]-self.demand_range[0]
            if avg_demand_range:
                self.minavg_demand = avg_demand_range[0]
                self.maxavg_demand = avg_demand_range[1]
        else:
            self.demand_range_value = [self.demand_range[prod][1]-self.demand_range[prod][0] for prod in range(num_products)]
            self.minavg_demand = [None]*num_products
            self.maxavg_demand = [None]*num_products
            for prod in range(num_products):
                if avg_demand_range[prod]:
                    self.minavg_demand[prod] = avg_demand_range[prod][0]
                    self.maxavg_demand[prod] = avg_demand_range[prod][1]

        # Não suporta demanda fixa (apenas para evitar ficar fazendo if toda hora 
        # para testar isso na hora de montar o estado)                        
        if not self.demand_config_by_product:
            assert self.demand_range[0] != self.demand_range[1]
        else:
            for prod in range(num_products):
                assert self.demand_range[prod][0] != self.demand_range[prod][1]

        if stochastic_leadtimes:
            # Os lead times são um para cada combinação produto/fornecedor, mais
            # um para cada link de transporte (nesse caso não é por produto porque
            # a capacidade de transporte é compartilhada)
            self.count_leadtimes_per_timestep = 0
            for node in self.nodes:            
                if node.num_supply_actions > 0:
                    self.count_leadtimes_per_timestep += self.num_products
                self.count_leadtimes_per_timestep += len(node.dests) if node.dests else 0

        # Definição dos espaços de ações e de estados
        action_space_size = 0
        for node in self.nodes:
            action_space_size += node.num_expected_actions()            

        # As observações (estados) são dados pela:
        # - len(self.last_level_nodes)      : 
        # - len(self.nodes)*1               : 
        # - 
        # - 1                               : 
        # ALTERAÇÃO: O estado agora tem informações de transporte de acordo com o momento que o material foi despachado.
        #            Isto aumentou o tamanho do estado, pois agora não há mais soma de material de períodos diferentes.
        obs_space_size = (len(self.last_level_nodes)*num_products       # Demandas, sendo 1 para cada varejista e produto
                          + len(self.nodes)*num_products                # Níveis de estoque, 1 para cada nó da cadeia e produto.
                          + len(self.nodes)*num_products*(max_leadtime) # Materiais em transporte; por produto, sendo l o leadtime máximo, considera, para cada nó
                                                                        # a quantidade de material em transporte que foi despachada nos períodos t, t-1, t-2,...,t-l+1
                          + 1)                                          # número de períodos para terminar o episódio

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


        # ALTERAÇÃO: O vetor de demanda guardava na posição t, a demanda para o período t+1. Agora ela guarda a demanda
        #            do próprio período t. Dessa forma, foi necessário incluir demandas zero para o período t=0.
        #            A última demanda gerada se torna inútil, mas foi mantida para que os experimentos anteriores continuassem
        #            compatíveis.

        # Se a configuração de demandas é a mesma pra todos os produtos        
        # Obs: o formato dos dados de demanda fica diferente nos dois casos, isso foi feito para
        #      manter compatibilidade com experimentos anteriores
        if not self.demand_config_by_product:
            
            # gerando as demandas de todo o episódio (por período/varejista/produto)
            
            self.customer_demands = np.concatenate((
                                       np.zeros((1, len(self.last_level_nodes), self.num_products)),
                                       generate_demand(self.rand_generator, 
                                                    (self.total_time_steps+1, len(self.last_level_nodes), self.num_products), 
                                                    self.total_time_steps, self.demand_range[0], self.demand_range[1],
                                                    std=self.demand_std, sen_peaks=self.demand_sen_peaks,
                                                    minavg=self.minavg_demand, maxavg=self.maxavg_demand,
                                                    perturb_norm=self.demand_perturb_norm)
                                       ))
        else: # se a configuração de demanda é separada por produto
            
            # gerando as demandas de todo o episódio (por produto/período/varejista)
            self.customer_demands = []
            for prod in range(self.num_products):
                self.customer_demands.append(
                        np.concatenate((
                                    np.zeros((1, len(self.last_level_nodes))),
                                    generate_demand(self.rand_generator, 
                                        (self.total_time_steps+1, len(self.last_level_nodes)),
                                        self.total_time_steps, self.demand_range[prod][0], self.demand_range[prod][1],
                                        std=self.demand_std[prod], sen_peaks=self.demand_sen_peaks[prod],
                                        minavg=self.minavg_demand[prod], maxavg=self.maxavg_demand[prod],
                                        perturb_norm=self.demand_perturb_norm[prod])
                            ))
                    )

        
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
        costs = {'stock':[], 'stock_pen':[], 'supply':[], 'process':[], 'process_pen':[], 'ship':[], 'ship_pen':[], 'unmet_dem':[]}
        units = {'stock':[], 'stock_pen':[], 'supply':[], 'process':[], 'process_pen':[], 'ship':[], 'ship_pen':[], 'unmet_dem':[]}
        
        # Zerando custos e unidades
        if self.build_info:
            for key in costs:
                costs[key] = [0]*self.num_products
            for key in units:
                units[key] = [0]*self.num_products

        return {'rewards':0, 'costs':costs, 'units':units}
        
    def _denormalize_action(self, action):
        return (action+1)/2
        
    def _normalize_obs(self, obs):
        return np.clip(obs*2 - 1, self.observation_space.low, self.observation_space.high)

    def step(self, action):
        action = self._denormalize_action(action)

        self.time_step += 1        

        total_cost = 0

        next_action_idx = 0
        next_leadt_idx  = 0
        next_customer = 0
        
        for node in self.nodes:
            
            actions_to_apply   = action[next_action_idx:next_action_idx+node.num_expected_actions()]
            next_action_idx   += node.num_expected_actions()

            if self.stochastic_leadtimes:
                num_leadtime_values = node.num_supply_actions + node.num_ship_actions//self.num_products
                leadtimes_to_apply = self.leadtimes[self.time_step-1, next_leadt_idx:next_leadt_idx+num_leadtime_values]
                next_leadt_idx += num_leadtime_values
            else:
                leadtimes_to_apply = node.num_expected_actions()*[self.avg_leadtime]            
            
            if node.last_level:
                # ALTERAÇÃO: A demanda do período t agora está na posição t. Antes estava na posição t-1
                if not self.demand_config_by_product:
                    demand = self.customer_demands[self.time_step, next_customer]
                else:
                    demand = [self.customer_demands[prod][self.time_step, next_customer] for prod in range(self.num_products)]
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

        return (self.current_state, self.current_reward, is_terminal, self.current_info)

    def _update_statistics(self):
        self.est_episode['rewards'] += self.current_reward
        costs = self.est_episode['costs']
        units = self.est_episode['units']
        for node in self.nodes:
            for key,value in node.est_costs.items():
                for prod in range(self.num_products):
                    costs[key][prod] += value[prod]
            for key,value in node.est_units.items():
                for prod in range(self.num_products):
                    units[key][prod] += value[prod]

    def _build_observation(self):
        """ Uma observação será formada:
            - Pelas demandas dos clientes
            - Pela obervação de cada nó. Que tem:
                - A quantidade de material em estoque.
                - O quanto de material está chegando nos períodos seguintes.
        """ 
        # Primeiro guardamos as demandas (normalizadas)
        # ALTERAÇÃO: Apesar do código aqui não ser alterado, a posição no vetor do período atual indica agora as demandas realizadas 
        #            no próprio período atual, e não mais as demandas para o próximo período.
        if not self.demand_config_by_product:
            demands_obs = (self.customer_demands[self.time_step,:].flatten() - self.demand_range[0])/(self.demand_range_value)
        else:
            demands_obs = []
            for n in range(len(self.last_level_nodes)):
                demands_obs.append([(self.customer_demands[prod][self.time_step,n].flatten() - self.demand_range[prod][0])/(self.demand_range_value[prod])
                                   for prod in range(self.num_products)])
            demands_obs = np.array(demands_obs).flatten()
            
        # Depois pegamos os dados de cada nó
        nodes_obs = []
        for node in self.nodes:
            nodes_obs += node.build_observation(self.time_step)

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
        # ALTERAÇÃO: Agora são exibidas as demandas atuais e não mais as demandas do próximo período
        if not self.demand_config_by_product:
            print('Current demands  :', self.customer_demands[self.time_step,:])
        else:
            print('Current demands (by prod) :', [self.customer_demands[prod,self.time_step,:] for prod in range(self.num_products)])
        print('Current state :', self.current_state)
        print('Current reward:', round(self.current_reward,3))
        print('='*30)

    def seed(self, seed=None):
        self.rand_generator = np.random.RandomState(seed)
        self.action_space.seed(0)

if __name__ == '__main__':
    num_products     = 1
    demand_range     = (10,20)
    stock_capacity   = 300
    ship_capacity    = 300
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
    exceeded_ship_capacity_cost = exceeded_capacity_cost
    
    total_time_steps = 5
    
    nodes_info = {}
    nodes_info['Supplier 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Supplier 2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'supply_capacity':supply_capacity, 'supply_cost':supply_cost,
                                'destinations':['Factory  1', 'Factory  2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Factory  1'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Factory  2'] = {'initial_stock':0, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'processing_capacity':processing_capacity, 'processing_cost':processing_cost,
                                'destinations':['Wholesal 1', 'Wholesal 2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Wholesal 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Wholesal 2'] = {'initial_stock':15, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'destinations':['Retailer 1', 'Retailer 2'], 'dest_costs':[[dest_cost]*2]*num_products, 
                                'ship_capacity':[ship_capacity]*2}
    nodes_info['Retailer 1'] = {'initial_stock':10, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}
    nodes_info['Retailer 2'] = {'initial_stock':20, 'stock_capacity':stock_capacity, 'stock_cost':stock_cost,
                                'last_level':True}

    env = SupplyChainEnv(nodes_info, num_products=num_products, demand_range=demand_range, unmet_demand_cost=unmet_demand_cost, 
                         exceeded_stock_capacity_cost=exceeded_capacity_cost, exceeded_process_capacity_cost=exceeded_capacity_cost,
                         exceeded_ship_capacity_cost=exceeded_ship_capacity_cost,
                         processing_ratio=processing_ratio, 
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

