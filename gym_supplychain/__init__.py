from gym.envs.registration import register

register(
    id='beergame-v0',
    entry_point='gym_supplychain.envs:BeerGameEnv',
)

register(
    id='beergame-v2',
    entry_point='gym_supplychain.envs:BeerGameEnv2',
)

register(
    id='supplychain-v0',
    entry_point='gym_supplychain.envs:SupplyChainEnv',
)

register(
    id='supplychain-ntom-v0',
    entry_point='gym_supplychain.envs:SupplyChainNtoMEnv',
)

register(
    id='supplychain-oneonen-v0',
    entry_point='gym_supplychain.envs:SupplyChainOneOneNEnv',
)

register(
    id='supplychain-linear-v0',
    entry_point='gym_supplychain.envs:SupplyChainLinearEnv',
)

register(
    id='supplychain-2perstage-v0',
    entry_point='gym_supplychain.envs:SupplyChain2perStageEnv',
)
