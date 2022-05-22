from dataclasses import dataclass, field
from typing import TypeVar

import clingo
import clingo.ast

from starlingo.Symbol import Symbol, Function
from starlingo.util import typecheck

ForwardAtom = TypeVar('ForwardAtom', bound='Atom')


@dataclass(frozen=True, order=True)
class Atom:
    symbol: Symbol = field(default_factory=Function)

    def __str__(self) -> str:
        return str(self.symbol)

    @staticmethod
    def from_clingo_symbol(symbol: clingo.Symbol) -> ForwardAtom:
        typecheck(symbol, clingo.SymbolType.Function, 'type')
        return Atom(Symbol.from_clingo_symbol(symbol))

    @classmethod
    def from_ast(cls, atom: clingo.ast.AST) -> ForwardAtom:
        typecheck(atom, clingo.ast.ASTType.SymbolicAtom, 'ast_type')
        symbol = Symbol.from_ast(atom.symbol)
        return Atom(symbol)
