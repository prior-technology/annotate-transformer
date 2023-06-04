"""Annotate Transformer extension for SymPy.

This module defines Symbols which represent intermediate calculations
in the Transformer model. These symbols are used to generate LaTeX
representations following the conventions described in the documentation
"""
from sympy import Symbol
class ResidualVector(Symbol):
    """Represents a residual vector in the Transformer as a 1-column matrix
    Generates a representation r_i^j for the ith token in the jth layer 
    """
    def __new__(cls, cache, token, layer):
        return Symbol('r_{%s}^{%s}' % (token, layer))
    

