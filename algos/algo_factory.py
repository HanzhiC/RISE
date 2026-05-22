# from algos.action_predictor import ActionDiffusionModule
# from algos.state_predictor import StateDiffusionModule
# from algos.state_action_predictor import StateActionDiffusionModule
# from algos.vlm_action import VLMActionModule
from algos.vla_wm_predictor import VLAWorldModelModule
from algos.vl_value_predictor import VLValuePredictorModule

def algorithm_factory(config):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.ALGORITHM
    train_config = config.TRAIN
    algo_name = algo_config.name

    # if algo_name == "action":
    #     algo = ActionDiffusionModule(algo_config=algo_config, train_config=train_config)
    # elif algo_name == "state":
    #     algo = StateDiffusionModule(algo_config=algo_config, train_config=train_config)
    # elif algo_name == "stateaction":
    #     algo = StateActionDiffusionModule(
    #         algo_config=algo_config, train_config=train_config
    #     )
    # elif algo_name == "vlm_action":
    #     algo = VLMActionModule(algo_config=algo_config, train_config=train_config)
    if algo_name == "vl_action_dynamics":
        algo = VLAWorldModelModule(algo_config=algo_config, train_config=train_config)
    elif algo_name == "vl_value_predictor":
        algo = VLValuePredictorModule(algo_config=algo_config, train_config=train_config)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo
