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
    id='sc-2perstage-v0',
    entry_point='gym_supplychain.envs:SupplyChain2perStageEnv',
)

register(
    id='sc-2perstage-seasonal-v0',
    entry_point='gym_supplychain.envs:SupplyChain2perStageSeasonalEnv',
)

register(
    id='sc-2perstage-multiproduct-v0',
    entry_point='gym_supplychain.envs:SupplyChainMultiProduct',
)

register(
    id='sc-Nperstage-multiproduct-v0',
    entry_point='gym_supplychain.envs:SupplyChainNPerStage',
)

register(
    id='sc-2perstage-multiproduct-inccosts-v0',
    entry_point='gym_supplychain.envs:SupplyChainMultiProduct_IncreasingCosts',
)

register(
    id='sc-2perstage-multiproduct-v1',
    entry_point='gym_supplychain.envs:SupplyChainMultiProduct_DemConfigByProd',
)

register(
    id='sc-2perstage-multiproduct-inccosts-v1',
    entry_point='gym_supplychain.envs:SupplyChainMultiProduct_DemConfigByProd_IncCosts',
)
