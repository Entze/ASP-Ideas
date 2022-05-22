import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TypeVar, Sequence

import clingo.ast

from starlingo.Atom import Atom
from starlingo.Symbol import Symbol
from starlingo.util import typecheck

ForwardLiteral = TypeVar('ForwardLiteral', bound='Literal')
ForwardBasicLiteral = TypeVar('ForwardBasicLiteral', bound='BasicLiteral')
ForwardConditionalLiteral = TypeVar('ForwardConditionalLiteral', bound='ConditionalLiteral')


class Sign(IntEnum):
    NoSign = 0
    DefaultNeg = 1


class Literal:
    def is_neg(self) -> bool:
        return NotImplemented

    def is_pos(self) -> bool:
        return NotImplemented

    @classmethod
    def from_ast(cls, literal: clingo.ast.AST) -> ForwardLiteral:
        if literal.ast_type is clingo.ast.ASTType.Literal:
            return BasicLiteral.from_ast(literal)
        elif literal.ast_type is clingo.ast.ASTType.ConditionalLiteral:
            return ConditionalLiteral.from_ast(literal)


@dataclass(frozen=True, order=True)
class BasicLiteral(Literal):
    atom: Atom = field(default_factory=Atom)
    sign: Sign = Sign.NoSign

    def is_neg(self) -> bool:
        return self.sign is Sign.DefaultNeg

    def is_pos(self) -> bool:
        return self.sign is Sign.NoSign

    def __str__(self):
        if self.sign is Sign.DefaultNeg:
            return "not {}".format(self.atom)
        return str(self.atom)

    def __abs__(self):
        return BasicLiteral(sign=Sign.NoSign, atom=copy.deepcopy(self.atom))

    def __neg__(self):
        return BasicLiteral(sign=Sign(self.sign ^ 1), atom=self.atom)

    def __invert__(self):
        return BasicLiteral(sign=Sign(self.sign ^ 1), atom=self.atom)

    @classmethod
    def from_ast(cls, literal: clingo.ast.AST) -> ForwardBasicLiteral:
        typecheck(literal, clingo.ast.ASTType.Literal, 'ast_type')
        sign = literal.sign
        atom = Atom.from_ast(literal.atom)
        return BasicLiteral(sign=Sign(sign), atom=atom)


@dataclass(frozen=True, order=True)
class ConditionalLiteral(Literal):
    literal: BasicLiteral = field(default_factory=BasicLiteral)
    condition: Sequence[Symbol] = ()

    def is_pos(self) -> bool:
        return True

    def is_neg(self) -> bool:
        return False

    def __str__(self):
        if self.condition:
            return "{}: {}".format(self.literal, ', '.join(map(str, self.condition)))
        else:
            return str(self.literal)

    @classmethod
    def from_ast(cls, conditional_literal: clingo.ast.AST) -> ForwardConditionalLiteral:
        typecheck(conditional_literal, clingo.ast.ASTType.ConditionalLiteral, 'ast_type')
        basic_literal = BasicLiteral.from_ast(conditional_literal.literal)
        condition = tuple(Symbol.from_ast(cond) for cond in conditional_literal.condition)
        return ConditionalLiteral(basic_literal, condition)
