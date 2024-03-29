# annotate-transformer

This repo will contain code for analysing and annotating the internal representations of natural language text processed through a generative transformer model.


## Definitions

Transformer Layer - a transformer block including attention, mlp, and normalization. Takes as input a point in embedding space form a lower layer, and pasts (points in embedding space for (each layer or the same layer?) for each previous position).

Token - an element of the transformer model vocabulary representing some part of the plaintext input to the LM or generated by the LM.

Position - the position of a token within the tokenized input

Linguistic Element - a representation of a self contained entity at some Transformer Layer. This is expected to include words and phrases at the initial level (i.e. straight after embedding the original token_ids).


## Questions

Is attention in a particular layer applied over all layers of previous positions, or only the same layer or lower layers?

Can we find a way to link tokens in a word - either at the lowest layer (after embedding) or highest (before decoding)
## Approach

Attribute the generate output to lower layers or earlier positions. Call each attributed point a linguistic-element. Linguistic elements in earlier positions are Closed. 


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
