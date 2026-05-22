from data.data_modules import Egoasis4DDataModule

def datamodule_factory(config):
    """
    A factory for creating pl.DataModule.

    Args:
        cls_name (str): name of the datamodule class
        config (Config): a config object

    Returns:
        A DataModule
    """
    # if config.ALGORITHM.name in ["action", "state", "stateaction", "vlm_action"]:
    #     datamodule = HOI4DDataModule(data_config=config.DATA, train_config=config.TRAIN)
    # else:
    #     raise NotImplementedError(
    #         "Algorithm {} is not a supported datamodule type".format(
    #             config.ALGORITHM.name
    #         )
    #     )
    datamodule = Egoasis4DDataModule(data_config=config.DATA, train_config=config.TRAIN)
    return datamodule
