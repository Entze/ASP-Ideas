import abc
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Optional, Sequence, Union

import clingo.ast


class Symbol(abc.ABC):

    @property
    @abc.abstractmethod
    def has_variable(self) -> bool:
        raise NotImplementedError


class TopLevelSymbol(Symbol, abc.ABC):

    @property
    def signature(self) -> str:
        return "{}/{}.".format(self.function_name, self.arity)

    @property
    @abc.abstractmethod
    def function_name(self) -> str:
        raise NotImplementedError

    @property
    def arity(self) -> int:
        return len(self.function_arguments)

    @property
    @abc.abstractmethod
    def function_arguments(self) -> Sequence[Symbol]:
        raise NotImplementedError


@dataclass(order=True, frozen=True)
class StringConstant:
    string: str = field(default="")

    def __str__(self):
        return '"{}"'.format(self.string)


@dataclass(order=True, frozen=True)
class IntegerConstant:
    number: int = field(default=0)

    def __str__(self):
        return str(self.number)


@dataclass(order=True, frozen=True)
class Term(Symbol):
    constant: Union[StringConstant, IntegerConstant] = field(default_factory=IntegerConstant)

    @property
    def has_variable(self) -> bool:
        return False


@dataclass(order=True, frozen=True)
class Function(TopLevelSymbol):
    name: Optional[str] = field(default=None)
    arguments: Sequence[Symbol] = field(default_factory=tuple)

    @cached_property
    def has_variable(self) -> bool:
        return any(argument.has_variable for argument in self.arguments)

    @property
    def function_name(self) -> str:
        return self.name

    @property
    def function_arguments(self) -> Sequence[Symbol]:
        return self.arguments

    @property
    def arity(self) -> int:
        return len(self.arguments)

    def __str__(self):
        if self.name is None:
            return '({})'.format(','.join(map(str, self.arguments)))
        elif not self.arguments:
            return self.name
        else:
            return '{}({})'.format(self.name, ','.join(map(str, self.arguments)))


@dataclass(order=True, frozen=True)
class Variable(Symbol):
    name: str

    @property
    def has_variable(self) -> bool:
        return True

    def __str__(self):
        return self.name


@dataclass(order=True, frozen=True)
class Atom:
    symbol: TopLevelSymbol = field(default_factory=Function)

    @property
    def has_variable(self) -> bool:
        return self.symbol.has_variable

    @property
    def signature(self) -> str:
        return self.symbol.signature

    def __str__(self):
        return str(self.symbol)


class Literal(abc.ABC):
    def __neg__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError

    @property
    def is_pos(self) -> bool:
        raise NotImplementedError

    @property
    def is_neg(self) -> bool:
        raise NotImplementedError


class Sign(IntEnum):
    NoSign = 0
    Negation = 1

    def __str__(self):
        if self is Sign.NoSign:
            return ''
        elif self is Sign.Negation:
            return 'not'
        else:
            assert False, 'Unknown IntEnum {} = {}.'.format(self.name, self.value)


@dataclass(order=True, frozen=True)
class BasicLiteral(Literal):
    sign: Sign = Sign.NoSign
    atom: Atom = field(default_factory=Atom)

    @property
    def is_pos(self) -> bool:
        return self.sign is Sign.NoSign

    @property
    def is_neg(self) -> bool:
        return self.sign is Sign.Negation

    @property
    def has_variable(self) -> bool:
        return self.atom.has_variable

    @property
    def atom_signature(self) -> str:
        return self.atom.signature

    def __str__(self):
        if self.sign is Sign.NoSign:
            return "{}".format(self.atom)
        else:
            return "{} {}".format(self.sign, self.atom)

    def __neg__(self):
        return BasicLiteral(Sign((self.sign ^ 1) % 2), self.atom)

    def __abs__(self):
        return BasicLiteral(Sign.NoSign, self.atom)


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


@dataclass(order=True, frozen=True)
class Comparison:
    left: Symbol = field(default_factory=Symbol)
    comparison: ComparisonOperator = field(default=ComparisonOperator.Equal)
    right: Symbol = field(default_factory=Symbol)

    def __str__(self):
        return "{}{}{}".format(self.left, self.comparison, self.right)


@dataclass(order=True, frozen=True)
class Rule(abc.ABC):
    head: Union[BasicLiteral, bool, None] = field(default=None)
    body: Optional[Sequence[BasicLiteral]] = field(default=None)

    @property
    def has_variables_in_head(self) -> bool:
        if isinstance(self, NormalRule):
            return self.head.has_variable
        return False

    @property
    def has_variables_in_body(self) -> bool:
        if self.body is not None:
            return any(literal.has_variable for literal in self.body)
        return False

    @property
    def has_variables(self) -> bool:
        return self.has_variables_in_head or self.has_variables_in_body

    @property
    @abc.abstractmethod
    def head_signature(self) -> str:
        raise NotImplementedError

    @staticmethod
    def fmt_body(body: Sequence[BasicLiteral]):
        return ', '.join(map(str, body))


@dataclass(order=True, frozen=True)
class NormalRule(Rule):
    head: BasicLiteral = field(default_factory=BasicLiteral)
    body: Sequence[BasicLiteral] = field(default_factory=tuple)

    @property
    def head_signature(self) -> str:
        return self.head.atom_signature

    def __str__(self):
        if self.body:
            return "{} :- {}.".format(self.head, Rule.fmt_body(self.body))
        else:
            return "{}.".format(self.head)

    def variable_normal_form(self):
        head_arguments = self.head.atom.symbol.function_arguments
        head_arguments = []
        comparisons = []
        v = 0
        stack = []
        for argument in head_arguments:
            v += 1
            var = Variable("_V{}".format(v))
            stack.append((argument, var))
            head_arguments.insert(0, var)
        while stack:
            symbol, var = stack.pop()
            comparison = Comparison(var, ComparisonOperator.Equal, symbol)
            comparisons.insert(0, comparison)
            if isinstance(symbol, TopLevelSymbol):
                arguments = symbol.function_arguments


@dataclass(order=True, frozen=True)
class IntegrityConstraint(Rule):
    body: Sequence[BasicLiteral] = field(default_factory=tuple)
    head: bool = field(default=False, init=False)

    @property
    def head_signature(self) -> str:
        return "#false/0."

    def __str__(self):
        if self.body:
            return '#false :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#false.'


@dataclass(order=True, frozen=True)
class Goal(Rule):
    body: Sequence[BasicLiteral] = field(default_factory=tuple)
    head: bool = field(default=True, init=False)

    @property
    def head_signature(self) -> str:
        return "#true/0."

    def __str__(self):
        if self.body:
            return '#true :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#true.'


@dataclass(order=True, frozen=True)
class Program:
    rules: Sequence[Rule] = field(default_factory=tuple)

    @cached_property
    def __canonical_program_dict(self):
        prop_atoms2rules = {}
        pred_atoms2rules = {}
        for rule in self.rules:
            maybe_prop = not rule.has_variables
            signature = rule.head_signature
            maybe_prop = maybe_prop and signature not in pred_atoms2rules

    @property
    def propositional_rules(self) -> Sequence[Rule]:
        pass

    @property
    def predicate_rules(self) -> Sequence[Rule]:
        pass

    def fmt(self, sep=' ', begin=None, end=None):
        b = begin + sep if begin is not None else ''
        e = sep + end if end is not None else ''
        return "{}{}{}".format(b, sep.join(map(str, self.rules)), e)

    @staticmethod
    def predicate_dual(predicate_rules: Sequence[Rule]):
        pass


A = Variable('A')
t_A_A = BasicLiteral(atom=Atom(Function(name='t', arguments=(A, A))))

r1 = NormalRule(head=t_A_A)

print(r1)
