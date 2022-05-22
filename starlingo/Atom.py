from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar

import clingo
import clingo.ast

from starlingo.Symbol import Symbol, Function, Term, SubSymbol
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


class ComparisonOperator(IntEnum):
    Equal = clingo.ast.ComparisonOperator.Equal.value
    GreaterEqual = clingo.ast.ComparisonOperator.GreaterEqual.value
    GreaterThan = clingo.ast.ComparisonOperator.GreaterThan.value
    LessEqual = clingo.ast.ComparisonOperator.LessEqual.value
    LessThan = clingo.ast.ComparisonOperator.LessThan.value
    NotEqual = clingo.ast.ComparisonOperator.NotEqual.value

    def __str__(self):
        if self is ComparisonOperator.Equal:
            op = '='
        elif self is ComparisonOperator.GreaterEqual:
            op = '>='
        elif self is ComparisonOperator.GreaterThan:
            op = '>'
        elif self is ComparisonOperator.LessEqual:
            op = '<='
        elif self is ComparisonOperator.LessThan:
            op = '<'
        else:
            assert self is ComparisonOperator.NotEqual, "Unknown ComparisonOperator {}".format(self)
            op = '!='
        return op


@dataclass(frozen=True, order=True)
class Comparison:
    left: SubSymbol = field(default_factory=Term)
    comparison: ComparisonOperator = field(default=ComparisonOperator.Equal)
    right: SubSymbol = field(default_factory=Term)

    def __str__(self):
        return "{}{}{}".format(self.left, self.comparison, self.right)

    @classmethod
    def from_ast(cls, comparison: clingo.ast.AST):
        typecheck(comparison, clingo.ast.ASTType.Comparison, 'ast_type')
        left = SubSymbol.from_ast(comparison.left)
        comparison_ = ComparisonOperator(comparison.comparison)
        right = SubSymbol.from_ast(comparison.right)
        return Comparison(left, comparison_, right)
