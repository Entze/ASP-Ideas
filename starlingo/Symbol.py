import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Sequence, TypeVar

import clingo
import clingo.ast

from starlingo.util import typecheck

ForwardSymbol = TypeVar('ForwardSymbol', bound='Symbol')
ForwardSubSymbol = TypeVar('ForwardSubSymbol', bound='SubSymbol')
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
    def from_ast(cls, symbol: clingo.ast.AST) -> ForwardSymbol:
        if symbol.ast_type is clingo.ast.ASTType.Function:
            return Function.from_ast(symbol)
        elif symbol.ast_type is clingo.ast.ASTType.UnaryOperation:
            return UnaryOperation.from_ast(symbol)
        elif symbol.ast_type in (
                clingo.ast.ASTType.Variable, clingo.ast.ASTType.SymbolicTerm, clingo.ast.ASTType.BinaryOperation):
            return SubSymbol.from_ast(symbol)
        else:
            assert False, "Unknown clingo.ast.ASTType {} of clingo.ast.AST {}.".format(symbol.ast_type, symbol)


class SubSymbol(Symbol):

    @classmethod
    def from_clingo_symbol(cls, symbol: clingo.Symbol) -> ForwardSubSymbol:
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

    @classmethod
    def from_ast(cls, symbol: clingo.ast.AST):
        if symbol.ast_type is clingo.ast.ASTType.Variable:
            return Variable.from_ast(symbol)
        elif symbol.ast_type is clingo.ast.ASTType.SymbolicTerm:
            return Term.from_ast(symbol)
        elif symbol.ast_type is clingo.ast.ASTType.BinaryOperation:
            return BinaryOperation.from_ast(symbol)
        elif symbol.ast_type in (clingo.ast.ASTType.Function, clingo.ast.ASTType.UnaryOperation):
            return Symbol.from_ast(symbol)
        else:
            assert False, "Unknown clingo.ast.ASTType {} of clingo.ast.AST {}.".format(symbol.ast_type, symbol)


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

    @classmethod
    def from_ast(cls, term: clingo.ast.AST):
        typecheck(term, clingo.ast.ASTType.SymbolicTerm, 'ast_type')
        return SubSymbol.from_clingo_symbol(term.symbol)


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

    def arguments_append(self, __object) -> ForwardUnaryOperator:
        argument = self.argument.arguments_append(__object)
        return UnaryOperation(self.operator, argument)

    @classmethod
    def from_ast(cls, unary_operation: clingo.ast.AST) -> ForwardSymbol:
        typecheck(unary_operation, clingo.ast.ASTType.UnaryOperation, 'ast_type')
        op = UnaryOperator(unary_operation.operator_type)
        argument = SubSymbol.from_ast(unary_operation.argument)
        return UnaryOperation(op, argument)


class BinaryOperator(IntEnum):
    Plus = clingo.ast.BinaryOperator.Plus.value
    Minus = clingo.ast.BinaryOperator.Minus.value

    def __str__(self):
        if self is BinaryOperator.Plus:
            return '+'
        elif self is BinaryOperator.Minus:
            return '-'
        else:
            assert False, "Unknown BinaryOperator {}: {}.".format(self.name, self.value)


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

        return "{}{}{}".format(left_str, self.operator, right_str)

    @classmethod
    def from_ast(cls, bin_op: clingo.ast.AST):
        typecheck(bin_op, clingo.ast.ASTType.BinaryOperation, 'ast_type')
        left = SubSymbol.from_ast(bin_op.left)
        operator = BinaryOperator(bin_op.operator_type)
        right = SubSymbol.from_ast(bin_op.right)
        return BinaryOperation(left, operator, right)


@dataclass(frozen=True, order=True)
class Function(Symbol):
    name: Optional[str] = None
    arguments: Sequence[SubSymbol] = ()

    @property
    def arity(self) -> int:
        return len(self.arguments)

    def __str__(self):
        if (self.name is None or self.name == '') and not self.arguments:
            return "()"
        elif (self.name is not None or self.name != '') and not self.arguments:
            return self.name
        elif (self.name is None or self.name == '') and self.arguments:
            return "({})".format(','.join(map(str, self.arguments)))
        else:
            return "{}({})".format(self.name, ','.join(map(str, self.arguments)))

    def match(self, name: Optional[str], arity: int = 0) -> bool:
        return name == self.name and arity == len(self.arguments)

    def match_signature(self, other: ForwardFunction) -> bool:
        return self.match(other.name, other.arity)

    def arguments_append(self, __object) -> ForwardFunction:
        arguments = list(self.arguments)
        arguments.append(__object)
        return Function(self.name, arguments)

    @classmethod
    def from_ast(cls, fun: clingo.ast.AST):
        typecheck(fun, clingo.ast.ASTType.Function, 'ast_type')
        name = fun.name
        arguments = tuple(Symbol.from_ast(argument) for argument in fun.arguments)
        return Function(name, arguments)
