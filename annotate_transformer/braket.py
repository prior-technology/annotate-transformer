"""Bra-Ket Notation
Bra-Ket notation is used in quantum mechanics to represent states and measurements, combined they represent the probability of a measurement of a state.  <x|ψ>  is the probability of measuring the state  |ψ>  as  x .

The transformer is understood as a performing a transformation on the residual space.

A sequence of tokens combine with the transformer to generate an output residual, which through the unembedding layer generates a weight representing likelihood of each of the next possible tokens.

The embedding and unembedding layers give a natural labelling for certain vectors in the residual space.

The input residual based on the token "the" is labelled as  the . The input residual from combining the unembedding vector with the position vector for the  ith  token is labelled as  thei–––– . The input residual from combining the unembedding vector with the position vector for the  ith  token and the position vector for the  jth  token is labelled as  theicatj–––––––– .

The vector from the unembedding layer representing the token "cat" is labelled as  cat¯¯¯¯¯¯¯ .

The transformer is represented as operator  T . Transformers are considered as an operation acting on the last residual stream.  T\ketThe is therefore the output residual from running the transformer on the single token " The".  \bracat¯¯¯¯¯¯¯T\ketThe––––  is the value of the output logit element for the token " cat". A transformer with tokens already in the context window is represented as  T(The cat is––––––––––) .
"""

def bra(x):
    return r'\bra{' + x + r'}'

def ket(x):
    return r'\ket{' + x + r'}'

def braket(x, y):
    return bra(x) + ket(y)

def underline(x):
    return r'\underline{' + x + r'}'

def sup(x):
    return r'^{' + x + r'}'

def token_vector(str_token, layer):
    return underline(str_token) + sup(layer)

def token_ket(str_token, layer):
    return ket(token_vector)

def mlp_operator(layer):
    return r'm^{' + layer + r'}'

def attention_head_operator(layer, head):
    return r'h^{' + layer + r'}_{' + head + r'}'

def attention_head_vector(layer, head, token):
    return attention_head_operator(layer, head) + token_vector(token, layer)

def expand_transformer_block(str_token, block_num):  
    last_residual = token_vector(str_token, block_num - 1)
    mlp_vector = mlp_operator(block_num) + token_vector(str_token, block_num-1)
    attention_vectors = r'\sum_{{j=0}}^6' + ket(attention_head_vector(block_num, 'j', str_token))
    return last_residual + mlp_vector + attention_vectors


