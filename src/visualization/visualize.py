import torch
def text_to_input_ids(tokenizer, text):
    toks = tokenizer.encode(text)
    return torch.as_tensor(toks).view(1, -1).cuda()

#from Ecco
def _one_hot(token_ids, vocab_size):
    return torch.zeros(len(token_ids), vocab_size).scatter_(1, token_ids.unsqueeze(1), 1.)

def _get_embeddings(model, input_ids):
    """
    Get token embeddings and one-hot vector into vocab. It's done via matrix multiplication
    so that gradient attribution is available when needed.
    Args:
        input_ids: Int tensor containing token ids. Of length (sequence length).
        Generally returned from the the tokenizer such as
        lm.tokenizer(text, return_tensors="pt")['input_ids'][0]
    Returns:
        inputs_embeds: Embeddings of the tokens. Dimensions are (sequence_len, d_embed)
        token_ids_tensor_one_hot: Dimensions are (sequence_len, vocab_size)
    """
    embedding_matrix = model.transformer.wte.weight
    vocab_size = embedding_matrix.shape[0]
    one_hot_tensor = _one_hot(input_ids, vocab_size).to('cuda')

    token_ids_tensor_one_hot = one_hot_tensor.clone().requires_grad_(True)
    # token_ids_tensor_one_hot.requires_grad_(True)

    inputs_embeds = torch.matmul(token_ids_tensor_one_hot, embedding_matrix)
    return inputs_embeds, token_ids_tensor_one_hot