from models.stateact.fuser import MultiModalFusionTransformerLite
import torch


@torch.no_grad()
def test_fuser():
    fuser = MultiModalFusionTransformerLite()
    x_obs = torch.randn(1, 196, 15, 384)
    x_action = torch.randn(1, 15, 384)
    x_language = torch.randn(1, 30, 384)
    state_tokens, action_tokens = fuser(x_obs, x_action, x_language)
    print(state_tokens.shape)
    print(action_tokens.shape)


if __name__ == "__main__":
    test_fuser()