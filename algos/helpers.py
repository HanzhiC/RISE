import torch


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        with torch.no_grad():
            ema_state_dict = ma_model.state_dict()
            for key, value in current_model.state_dict().items():
                ema_value = ema_state_dict[key]
                ema_value.copy_(self.beta * ema_value + (1.0 - self.beta) * value)
