from models.multimodal.t5_encoder import T5Embedder
import torch

tokenizer_max_length = 1024

device = torch.device(f"cuda:0")
text_embedder = T5Embedder(
    from_pretrained="google-t5/t5-small",
    model_max_length=1024,
    device=device,
    use_offload_folder=None,
)
tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model


INSTRUCTION = "Pick."

tokens = tokenizer(
    INSTRUCTION, return_tensors="pt", padding="longest", truncation=True
)["input_ids"].to(device)

tokens = tokens.view(1, -1)
with torch.no_grad():
    pred = text_encoder(tokens).last_hidden_state.detach().cpu()

breakpoint()