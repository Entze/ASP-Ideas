import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence, TypeVar

import clingo
import clingo.ast

from starlingo.util import typecheck

ForwardSymbol = TypeVar('ForwardSymbol', bound='Symbol')
ForwardFunction = TypeVar('ForwardFunction', bound='Function')
ForwardUnaryOperator = TypeVar('ForwardUnaryOperator', bound='UnaryOperator')
ForwardBinaryOperator = TypeVar('ForwardBinaryOperator', bound='BinaryOperator')


class Symbol:

    def __neg__(self):
        return UnaryOperation(UnaryOperator.Minus, copy.deepcopy(self))

    def __add__(self, other):
        return BinaryOperation(copy.deepcopy(self), BinaryOperator.Plus, other)

    def is_function(self) -> bool:
        return isinstance(self, Function)

    def is_unary_operation(self) -> bool:
        return isinstance(self, UnaryOperation)

    def is_binary_operation(self) -> bool:
        return isinstance(self, BinaryOperation)

    def is_operation(self) -> bool:
        return self.is_unary_operation() or self.is_binary_operation()

    def is_variable(self) -> bool:
        return isinstance(self, Variable)

    def is_term(self) -> bool:
        return isinstance(self, Term)

    @classmethod
    def from_clingo_symbol(cls, symbol: clingo.Symbol) -> ForwardSymbol:
        if symbol.type is clingo.SymbolType.Function:
            name: str = symbol.name
            arguments = tuple(SubSymbol.from_clingo_symbol(argument) for argument in symbol.arguments)
            return Function(name, arguments)
        else:
            assert False, "Unknown clingo.SymbolType {}.".format(symbol.type)

    @classmethod
    def from_ast(cls, symbol: clingo.ast.AST):
        if symbol.ast_type is clingo.ast.ASTType.Function:
            return Function.from_ast(symbol)
        elif symbol.ast_type is clingo.ast.ASTType.Variable:
            return Variable.from_ast(symbol)
        elif symbol.ast_type is clingo.ast.ASTType.Comparison:
            return Comparison.from_ast(symbol)


class SubSymbol(Symbol):

    @classmethod
    def from_clingo_symbol(cls, symbol: clingo.Symbol) -> Symbol:
        if symbol.type is clingo.SymbolType.Number:
            return Term(IntegerConstant(symbol.number))
        elif symbol.type is clingo.SymbolType.String:
            return Term(StringConstant(symbol.string))
        elif symbol.type is clingo.SymbolType.Function:
            f_symbol = Symbol.from_clingo_symbol(symbol)
            if symbol.negative:
                return UnaryOperation(UnaryOperator.Minus, f_symbol)
            return f_symbol
        else:
            assert False, "Unknown clingo.SymbolType {}.".format(symbol.type)


@dataclass(frozen=True, order=True)
class Variable(SubSymbol):
    name: str

    def __str__(self):
        return self.name

    @classmethod
    def from_ast(cls, variable: clingo.ast.AST):
        typecheck(variable, clingo.ast.ASTType.Variable, 'ast_type')
        return Variable(variable.name)


class Constant:
    pass


@dataclass(frozen=True, order=True)
class StringConstant(Constant):
    string: str = ""

    def __str__(self):
        return '"{}"'.format(self.string)


@dataclass(frozen=True, order=True)
class IntegerConstant(Constant):
    number: int = 0

    def __str__(self):
        return str(self.number)


@dataclass(frozen=True, order=True)
class Term(SubSymbol):
    constant: Constant = field(default=IntegerConstant())

    def __str__(self):
        return str(self.constant)


class UnaryOperator(IntEnum):
    Minus = clingo.ast.UnaryOperator.Minus.value


@dataclass(frozen=True, order=True)
class UnaryOperation(Symbol):
    operator: UnaryOperator
    argument: SubSymbol

    def __str__(self) -> str:
        if self.operator is UnaryOperator.Minus:
            if self.argument.is_operation():
                return "-({})".format(self.argument)
            return "-{}".format(self.argument)
        else:
            assert False, "Unknown UnaryOperatorType {}.".format(self.operator)


class BinaryOperator(IntEnum):
    Plus = clingo.ast.BinaryOperator.Plus.value


@dataclass(frozen=True, order=True)
class BinaryOperation(SubSymbol):
    left: SubSymbol
    operator: BinaryOperator
    right: SubSymbol

    def __str__(self) -> str:
        left_str = str(self.left)
        if self.left.is_operation():
            left_str = '({})'.format(left_str)
        right_str = str(self.right)
        if self.right.is_operation():
            right_str = '({})'.format(right_str)

        if self.operator is BinaryOperator.Plus:
            return "{}+{}".format(left_str, right_str)


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
class Comparison(Symbol):
    left: SubSymbol = field(default_factory=Term)
    operator: ComparisonOperator = field(default=ComparisonOperator.Equal)
    right: SubSymbol = field(default_factory=Term)

    def __str__(self):
        return "{}{}{}".format(self.left, self.operator, self.right)

    @classmethod
    def from_ast(cls, comparison: clingo.ast.AST):
        typecheck(comparison, clingo.ast.ASTType.Comparison, 'ast_type')
        left = SubSymbol.from_ast(comparison.left)
        operator = ComparisonOperator(comparison.operator)
        right = SubSymbol.from_ast(comparison.right)
        return Comparison(left, operator, right)


@dataclass(frozen=True, order=True)
class Function(Symbol):
    name: Optional[str] = None
    arguments: Sequence[SubSymbol] = ()

    @property
    def arity(self) -> int:
        return len(self.arguments)

    def __str__(self):
        if self.name is None and not self.arguments:
            return "()"
        elif self.name is not None and not self.arguments:
            return self.name
        elif self.name is None and self.arguments:
            return "({})".format(','.join(map(str, self.arguments)))
        else:
            return "{}({})".format(self.name, ','.join(map(str, self.arguments)))

    def match(self, name: Optional[str], arity: int = 0) -> bool:
        return name == self.name and arity == len(self.arguments)

    def match_signature(self, other: ForwardFunction) -> bool:
        return self.match(other.name, other.arity)

    @classmethod
    def from_ast(cls, fun: clingo.ast.AST):
        typecheck(fun, clingo.ast.ASTType.Function, 'ast_type')
        name = fun.name
        arguments = tuple(Symbol.from_ast(argument) for argument in fun.arguments)
        return Function(name, arguments)
