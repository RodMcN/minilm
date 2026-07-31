import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from src.minilm.tokeniser import END_MESSAGE_TOKEN


TEST_PROMPTS = [
    "Hello, my name is",
    "Once upon a time, there was",
    "The glass fell onto the floor and",
    "Anna gave the book to Leo. He put it",
    "Sarah was hungry, so she went to",
    "The dog heard a noise outside, so it",
    "Two plus three equals",
    "The capital of France is",
    "The sun rises in the",
    "Maya opened the small wooden box. Inside, she found",
]


@torch.no_grad()
def run_test_prompts(model, tokeniser, prompts=None, max_tokens=10):
    prompts = TEST_PROMPTS if prompts is None else prompts
    device = next(model.parameters()).device
    pad_id = tokeniser.token_to_id(END_MESSAGE_TOKEN)

    sequences = [torch.tensor(tokeniser.encode(p).ids, device=device) for p in prompts]
    lengths = torch.tensor([len(x) for x in sequences], device=device)
    ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
    rows = torch.arange(len(prompts), device=device)

    for _ in range(max_tokens):
        logits = model(ids)[0][rows, lengths - 1]
        next_ids = torch.distributions.Categorical(logits=logits).sample()

        ids = F.pad(ids, (0, 1), value=pad_id)
        ids[rows, lengths] = next_ids
        lengths += 1

    return [
        (prompts[i], str([tokeniser.decode([t]) for t in ids[i, :length].tolist()]))
        for i, length in enumerate(lengths.tolist())
    ]
