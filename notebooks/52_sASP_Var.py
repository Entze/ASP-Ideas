import abc
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Optional, Sequence, Union, Dict, List, Tuple, Set, TypeVar, Mapping, MutableSequence

import clingo.ast

ForwardSymbol = TypeVar('ForwardSymbol', bound='Symbol')
ForwardVariable = TypeVar('ForwardVariable', bound='Variable')


class Symbol(abc.ABC):

    @property
    @abc.abstractmethod
    def has_variable(self) -> bool:
        raise NotImplementedError

    @property
    def variables(self) -> Set[ForwardVariable]:
        variables = set()
        stack = [self]
        while stack:
            current = stack.pop()
            if isinstance(current, Variable):
                variables.add(current)
            elif isinstance(current, TopLevelSymbol):
                stack.extend(current.function_arguments)
        return variables

    @abc.abstractmethod
    def substitute_variables(self, substitute_map: Mapping[ForwardVariable, ForwardSymbol]) -> ForwardSymbol:
        raise NotImplementedError


@dataclass(order=True, frozen=True)
class Variable(Symbol):
    name: str

    @property
    def has_variable(self) -> bool:
        return True

    def __str__(self):
        return self.name

    def substitute_variables(self, substitute_map: Mapping[ForwardVariable, ForwardSymbol]) -> Symbol:
        if self in substitute_map:
            return substitute_map[self]
        return self


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

    def __str__(self):
        return str(self.constant)

    def substitute_variables(self, substitute_map: Mapping[ForwardVariable, ForwardSymbol]) -> Symbol:
        return self

    @staticmethod
    def zero():
        return Term(IntegerConstant(0))

    @staticmethod
    def one():
        return Term(IntegerConstant(1))


ForwardTopLevelSymbol = TypeVar('ForwardTopLevelSymbol', bound='TopLevelSymbol')


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

    def variable_normal_form_env(self, env: Optional[Dict[tuple, Variable]] = None) -> Dict[tuple, Variable]:
        if env is None:
            env = dict()
        queue: List[Tuple[Symbol, tuple]] = [(self, ())]
        witnessed_variables = set()
        v = -1
        while queue:
            v += 1
            current, pos = queue.pop(0)
            if isinstance(current, Variable) and current not in witnessed_variables:
                witnessed_variables.add(current)
                env[pos] = current
                continue
            var = Variable("V{}".format(v))
            env[pos] = var
            if isinstance(current, TopLevelSymbol):
                for i, argument in enumerate(current.function_arguments):
                    queue.append((argument, (*pos, i)))
        return env

    @abc.abstractmethod
    def variable_normal_form(self, env: Dict[Symbol, Variable], context: tuple = ()):
        raise NotImplementedError

    @abc.abstractmethod
    def substitute_variables(self, substitute_map: Mapping[ForwardVariable, ForwardSymbol]) -> ForwardTopLevelSymbol:
        raise NotImplementedError


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

    def variable_normal_form(self, env: Dict[tuple, Variable], context: tuple = ()):
        arguments = []
        for i, argument in enumerate(self.arguments):
            pos = (*context, i)
            assert env[pos] is not None or isinstance(argument, Variable), "Variable {} should have pos {}.".format(
                argument, pos)
            if pos not in env:
                arguments.append(argument)
            else:
                arguments.append(env[pos])
        return Function(self.name, tuple(arguments))

    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]) -> TopLevelSymbol:
        arguments = []
        for argument in self.arguments:
            if isinstance(argument, Variable) and argument in substitute_map:
                arguments.append(substitute_map[argument])
            elif isinstance(argument, TopLevelSymbol):
                arguments.append(argument.substitute_variables(substitute_map))
            else:
                arguments.append(argument)
        return Function(self.name, tuple(arguments))


ForwardAtom = TypeVar('ForwardAtom', bound='Atom')


@dataclass(order=True, frozen=True)
class Atom:
    symbol: TopLevelSymbol = field(default_factory=Function)

    @property
    def has_variable(self) -> bool:
        return self.symbol.has_variable

    @property
    def variables(self) -> Set[Variable]:
        return self.symbol.variables

    @property
    def signature(self) -> str:
        return self.symbol.signature

    def match(self, other: ForwardAtom) -> bool:
        return self.symbol.function_name == other.symbol.function_name and \
               len(self.symbol.function_arguments) == len(other.symbol.function_arguments)

    def __str__(self):
        return str(self.symbol)

    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]):
        return Atom(self.symbol.substitute_variables(substitute_map))


class ClauseElement(abc.ABC):

    @property
    @abc.abstractmethod
    def has_variable(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variables(self) -> Set[Variable]:
        raise NotImplementedError

    @abc.abstractmethod
    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def __abs__(self):
        raise NotImplementedError


class HeadClauseElement(ClauseElement):
    pass


class Literal(HeadClauseElement):

    @property
    @abc.abstractmethod
    def atom_signature(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def is_pos(self) -> bool:
        raise NotImplementedError

    @property
    @abc.abstractmethod
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
    def variables(self) -> Set[Variable]:
        return self.atom.variables

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

    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]):
        return BasicLiteral(self.sign, self.atom.substitute_variables(substitute_map))


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
class Comparison(ClauseElement):
    left: Symbol = field(default_factory=Symbol)
    comparison: ComparisonOperator = field(default=ComparisonOperator.Equal)
    right: Symbol = field(default_factory=Symbol)

    def __abs__(self):
        raise NotImplementedError

    def __neg__(self):
        if self.comparison is ComparisonOperator.Equal:
            return Comparison(self.left, ComparisonOperator.NotEqual, self.right)
        elif self.comparison is ComparisonOperator.NotEqual:
            return Comparison(self.left, ComparisonOperator.Equal, self.right)
        else:
            raise NotImplementedError

    def __str__(self):
        return "{}{}{}".format(self.left, self.comparison, self.right)

    @property
    def has_variable(self) -> bool:
        return self.left.has_variable or self.right.has_variable

    @property
    def variables(self) -> Set[Variable]:
        return self.left.variables | self.right.variables

    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]):
        return Comparison(self.left.substitute_variables(substitute_map), self.comparison,
                          self.right.substitute_variables(substitute_map))


ForwardDirective = TypeVar('ForwardDirective', bound='Directive')


@dataclass(order=True, frozen=True)
class Directive(HeadClauseElement):
    name: str
    arguments: Sequence[Union[ForwardDirective, Symbol]] = field(default_factory=tuple)

    def __abs__(self):
        raise NotImplementedError

    def __neg__(self):
        if self.name == 'true':
            return Directive.false()
        elif self.name == 'false':
            return Directive.true()
        else:
            raise NotImplementedError

    def __str__(self):
        if not self.arguments:
            return "#{}".format(self.name)
        else:
            return "#{}({})".format(self.name, ','.join(map(str, self.arguments)))

    @property
    def has_variable(self) -> bool:
        if self.name == 'forall':
            return True
        return any(argument.has_variable for argument in self.arguments)

    @property
    def variables(self) -> Set[Variable]:
        return set(variable for element in self.arguments for variable in element.variables)

    def substitute_variables(self, substitute_map: Mapping[Variable, Symbol]):
        arguments = tuple(argument.substitute_variables(substitute_map) for argument in self.arguments)
        return Directive(self.name, arguments)

    @staticmethod
    def forall(var: Variable, goal: Union[ForwardDirective, Symbol]):
        return Directive('forall', (var, goal,))

    @staticmethod
    def true():
        return Directive('true')

    @staticmethod
    def false():
        return Directive('false')


@dataclass(order=True, frozen=True)
class Rule(abc.ABC):
    head: Optional[HeadClauseElement] = field(default=None)
    body: Optional[Sequence[ClauseElement]] = field(default=None)

    @property
    def has_variables_in_head(self) -> bool:
        if isinstance(self, NormalRule):
            return self.head.has_variable
        return False

    @property
    def has_variables_in_body(self) -> bool:
        if self.body is not None:
            return any(element.has_variable for element in self.body)
        return False

    def variables_in_body(self) -> Set[Variable]:
        if self.body is not None:
            return set(variable for element in self.body for variable in element.variables)

    @property
    def has_variables(self) -> bool:
        return self.has_variables_in_head or self.has_variables_in_body

    @property
    @abc.abstractmethod
    def head_signature(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def variable_normal_form(self):
        raise NotImplementedError

    @staticmethod
    def fmt_body(body: Sequence[ClauseElement]):
        return ', '.join(map(str, body))


@dataclass(order=True, frozen=True)
class NormalRule(Rule):
    head: BasicLiteral = field(default_factory=BasicLiteral)
    body: Sequence[ClauseElement] = field(default_factory=tuple)

    @property
    def head_signature(self) -> str:
        return self.head.atom_signature

    @property
    def has_existential_vars(self) -> bool:
        if not self.has_variables_in_body:
            return False
        return self.head.variables != set(variable for element in self.body for variable in element.variables)

    def __str__(self):
        if self.body:
            return "{} :- {}.".format(self.head, Rule.fmt_body(self.body))
        else:
            return "{}.".format(self.head)

    def variable_normal_form(self, env: Optional[Dict[tuple, Variable]] = None):
        if env is None:
            env = dict()
        self.head.atom.symbol.variable_normal_form_env(env)
        assert env is not None
        stack: List[Tuple[Symbol, tuple]] = []
        substitute_map = dict()
        for i, argument in enumerate(self.head.atom.symbol.function_arguments):
            stack.append((argument, (i,)))
        equalities = []
        new_head = BasicLiteral(self.head.sign, Atom(self.head.atom.symbol.variable_normal_form(env)))
        for i, argument in enumerate(new_head.atom.symbol.function_arguments):
            var = Variable("V{}".format(i))
            if var != argument:
                assert isinstance(argument, Variable)
                substitute_map[argument] = var
        while stack:
            current, pos = stack.pop(0)
            assert pos in env or isinstance(current, Variable), "Variable {} should have a pos {}.".format(current, pos)
            if isinstance(current, TopLevelSymbol):
                target = current.variable_normal_form(env, pos)
                for i, argument in enumerate(current.function_arguments):
                    stack.append((argument, (*pos, i)))
            else:
                target = current
            if env[pos] != target:
                equalities.append(
                    Comparison(env[pos], ComparisonOperator.Equal, target).substitute_variables(substitute_map))
        new_body = (*equalities, *(element.substitute_variables(substitute_map) for element in self.body))
        new_rule = NormalRule(new_head.substitute_variables(substitute_map), new_body)
        return new_rule


@dataclass(order=True, frozen=True)
class IntegrityConstraint(Rule):
    body: Sequence[BasicLiteral] = field(default_factory=tuple)
    head: Directive = field(default=Directive.false(), init=False)

    @property
    def head_signature(self) -> str:
        return "#false/0."

    def __str__(self):
        if self.body:
            return '#false :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#false.'

    def variable_normal_form(self):
        return self


@dataclass(order=True, frozen=True)
class Goal(Rule):
    body: Sequence[BasicLiteral] = field(default_factory=tuple)
    head: Directive = field(default=Directive.true(), init=False)

    @property
    def head_signature(self) -> str:
        return "#true/0."

    def __str__(self):
        if self.body:
            return '#true :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#true.'

    def variable_normal_form(self):
        return self


ForwardCoinductiveHypothesesSet = TypeVar('ForwardCoinductiveHypothesesSet', bound='CoinductiveHypothesesSet')


def empty_constraints():
    return defaultdict(set)


@dataclass
class CoinductiveHypothesesSet:
    hypotheses: Set[BasicLiteral] = field(default_factory=set)
    bounds: Dict[Variable, Set[Symbol]] = field(default_factory=empty_constraints)
    negative_constraints: Dict[Variable, Set[Symbol]] = field(default_factory=empty_constraints)

    def __contains__(self, item):
        if isinstance(item, BasicLiteral):
            return item in self.hypotheses
        if isinstance(item, Variable):
            return self.bounds[item] or self.negative_constraints[item]

    def __str__(self):
        return self.fmt()

    def is_negativly_constrained(self, variable: Variable) -> bool:
        return variable in self.negative_constraints and bool(self.negative_constraints[variable])

    def is_bound(self, variable: Variable) -> bool:
        return variable in self.bounds and bool(self.bounds[variable])

    def fmt(self, sep=' ', literal_sep=',', variable_sep=' ', constraint_sep=' '):
        fmt = "{}{}{}".format('{', literal_sep.join(map(str, self.hypotheses)), '}', )
        fmt += sep
        for variable in self.bounds:
            fmt += variable_sep
            for bound in self.bounds[variable]:
                fmt += constraint_sep
                fmt += "{} = {}".format(variable, bound)
            if self.bounds[variable]:
                fmt += constraint_sep
        if self.bounds:
            fmt += variable_sep
        fmt += sep
        for variable in self.negative_constraints:
            fmt += variable_sep
            for negative_constraint in self.negative_constraints[variable]:
                fmt += constraint_sep
                fmt += "{} /= {}".format(variable, negative_constraint)
            if self.negative_constraints[variable]:
                fmt += constraint_sep
        if self.negative_constraints:
            fmt += variable_sep
        return fmt

    def bind_value(self, variable: Variable, value: Symbol) -> bool:
        if not isinstance(value, Variable) and value not in self.bounds[variable] and len(self.bounds[variable]) > 0:
            return False
        if value in self.negative_constraints[variable]:
            return False
        #if not isinstance(value, Variable) and any(constructive_unification(value, negative_constraint) for negative_constraint in self.negative_constraints[variable]):
        #    return False
        self.bounds[variable].add(value)
        return True

    def forbid_value(self, variable: Variable, value: Symbol) -> bool:
        if value in self.bounds[variable]:
            return False
        #if not isinstance(value, Variable) and any(constructive_unification(value, bound) for bound in self.bounds[variable]):
        #    return False
        self.negative_constraints[variable].add(value)
        return True

    def forbid_values(self, variable: Variable, *values: Symbol) -> bool:
        if not self.bounds[variable].isdisjoint(values):
            return False
        for value in values:
            c = self.forbid_value(variable, value)
            if not c:
                return False
        return True

    def positive_constraints(self, literal: BasicLiteral, variable: Variable) -> Set[Symbol]:
        pass

    def get_value(self, literal: BasicLiteral, variable: Variable) -> Optional[Symbol]:
        return next(iter(self.positive_constraints(literal, variable)), None)

    def negative_constraints(self, literal: BasicLiteral, variable: Variable) -> Set[Symbol]:
        pass

    def unify(self, src: BasicLiteral, dst: BasicLiteral, dst_chs: ForwardCoinductiveHypothesesSet) -> bool:
        if src.atom.symbol.function_name != dst.atom.symbol.function_name:
            return False
        nargs = len(src.atom.symbol.function_arguments)
        if nargs != len(dst.atom.symbol.function_arguments):
            return False
        for i in range(nargs):
            src_arg: Symbol = src.atom.symbol.function_arguments[i]
            dst_arg: Symbol = dst.atom.symbol.function_arguments[i]
            if isinstance(src_arg, Variable):
                src_val = self.get_value(src, src_arg)
            else:
                src_val = src_arg
            if src_val is not None:
                if isinstance(dst_arg, Variable):
                    dst_val = dst_chs.get_value(dst, dst_arg)
                    if dst_val is None:
                        unifiable = dst_chs.bind_value(dst, dst_arg, src_val)
                        if isinstance(src_arg, Variable):
                            unifiable = unifiable and dst_chs.forbid_values(dst, dst_arg,
                                                                            *self.negative_constraints(src, src_arg))
                        if not unifiable:
                            return False
                    elif src_val != dst_val:
                        return False
                elif dst_arg != src_val:
                    return False
        return True

    def constructive_unification(self, left: Symbol, right: Symbol) -> bool:
        pass


Forward_BaseNode = TypeVar('Forward_BaseNode', bound='_BaseNode')


@dataclass
class _BaseNode:
    subject: Union[ClauseElement, Rule, Directive]
    parent: Optional[Forward_BaseNode]
    children: Optional[Sequence[Forward_BaseNode]]
    index: int
    is_exhausted: bool
    hypotheses: Dict[BasicLiteral, Dict[Variable, Dict[str, Set[Symbol]]]]

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_success(self) -> bool:
        raise NotImplementedError

    @property
    def is_expanded(self) -> bool:
        return self.children is not None

    def expand(self, rule_dict: Mapping[str, Mapping[str, List[NormalRule]]]):
        raise NotImplementedError


@dataclass
class BaseNode(_BaseNode):
    subject: ClauseElement = field(default_factory=BasicLiteral)
    parent: Optional[_BaseNode] = field(default=None)
    children: Optional[Sequence[Forward_BaseNode]] = field(default=None)
    index: int = field(default=0)
    is_exhausted: bool = field(default=False)
    hypotheses: Dict[BasicLiteral, Dict[Variable, Dict[str, Set[Symbol]]]] = field(default_factory=dict)


ForwardOrNode = TypeVar('ForwardOrNode', bound='OrNode')
ForwardAndNode = TypeVar('ForwardAndNode', bound='AndNode')


@dataclass
class Leaf(BaseNode):
    subject: Directive = field(default_factory=Directive.false)
    parent: Optional[BaseNode] = field

    @property
    def is_success(self) -> bool:
        return self.subject.name == 'true'

    @property
    def is_expanded(self) -> bool:
        return True


@dataclass
class OrNode(BaseNode):
    subject: ClauseElement = field(default_factory=BasicLiteral)
    parent: Optional[ForwardAndNode] = field(default=None)
    children: Optional[MutableSequence[Union[ForwardAndNode, Leaf]]] = field(default=None)

    @property
    def is_success(self) -> bool:
        return self.is_expanded and any(child.is_success for child in self.children)

    def expand(self, rule_dict: Mapping[str, Mapping[str, List[NormalRule]]]):
        if self.children is None:
            self.children = []
        assert self.children is not None
        child = None
        if isinstance(self.subject, BasicLiteral):
            if self.subject in self.hypotheses:
                child = Leaf(subject=Directive.true(), parent=self)
            elif -self.subject in self.hypotheses:
                child = Leaf(subject=Directive.false(), parent=self)
            else:
                hypotheses = sorted(*self.hypotheses)
                if self.index < len(hypotheses):
                    pass

                # else:
                rules_ = rule_dict[self.subject.atom_signature]
                if self.subject.is_pos:
                    rules = rules_['primal']
                else:
                    rules = rules_['dual']
                if self.index >= len(rules):
                    self.is_exhausted = True
                else:
                    rule = rules[self.index]
                    child_hypotheses = dict()
                    #unifiable = unify(self.subject, self.hypotheses,
                    #                  rule.head, child_hypotheses)
                    #if unifiable:
                    #    child = AndNode(subject=rule,
                    #                    parent=self,
                    #                    hypotheses=child_hypotheses),

        elif isinstance(self.subject, Comparison):
            if self.subject.comparison.Equal:
                pass
            elif self.subject.comparison.NotEqual:
                pass
        elif isinstance(self.subject, Directive):
            if self.subject.name in ('true', 'false'):
                child = Leaf(self.subject, parent=self)

        if child is not None:
            self.children.append(child)
        self.index += 1
        return child


@dataclass
class AndNode(BaseNode):
    subject: Rule = field(default_factory=NormalRule)
    parent: Optional[OrNode] = field(default=None)
    children: Optional[MutableSequence[Union[OrNode, Leaf]]] = field(default=None)

    @property
    def is_success(self) -> bool:
        return self.is_expanded and all(child.is_success for child in self.children)

    def expand(self, rule_dict: Mapping[str, Mapping[str, List[NormalRule]]]):
        if self.children is None:
            self.children = []
        child = None
        if self.subject.body:
            if self.index < len(self.subject.body):
                element = self.subject.body[self.index]
                child = OrNode(subject=element,
                               parent=self,
                               hypotheses=deepcopy(self.hypotheses))
            else:
                self.is_exhausted = True
        else:
            child = Leaf(subject=Directive.true(), parent=self)
            self.is_exhausted = True

        if child is not None:
            self.children.append(child)
        self.index += 1
        return child


@dataclass(order=True, frozen=True)
class Program:
    rules: Sequence[Rule] = field(default_factory=tuple)

    @cached_property
    def program_dicts(self):
        prop_atoms2rules = defaultdict(set)
        pred_atoms2rules = defaultdict(set)
        queue: List[Rule] = [*self.rules]
        while queue:
            rule = queue.pop(0)
            maybe_prop = not rule.has_variables
            signature = rule.head_signature
            maybe_prop = maybe_prop and signature not in pred_atoms2rules
            if maybe_prop:
                prop_atoms2rules[signature].add(rule)
            else:
                pred_atoms2rules[signature].add(rule)
        return prop_atoms2rules, pred_atoms2rules

    @cached_property
    def canonical_program_dicts(self):
        prop_atoms2rules = defaultdict(set)
        pred_atoms2rules = defaultdict(set)
        queue: List[Rule] = [*self.rules]
        while queue:
            rule = queue.pop(0)
            maybe_prop = not rule.has_variables
            signature = rule.head_signature
            maybe_prop = maybe_prop and signature not in pred_atoms2rules
            if maybe_prop:
                prop_atoms2rules[signature].add(rule)
            else:
                rule_vnf = rule.variable_normal_form()
                pred_atoms2rules[signature].add(rule_vnf)
                if signature in prop_atoms2rules:
                    queue.extend(prop_atoms2rules[signature])
                    prop_atoms2rules[signature].clear()
                for literal in rule.body:
                    if not literal.has_variable or not isinstance(literal, Literal):
                        continue
                    if abs(literal) in prop_atoms2rules:
                        queue.extend(prop_atoms2rules[literal.atom_signature])
                        prop_atoms2rules[literal.atom_signature].clear()
                    else:
                        pred_atoms2rules[literal.atom_signature] = set()

        return prop_atoms2rules, pred_atoms2rules

    @cached_property
    def sASP_program_dict(self):
        signature_rules = defaultdict(lambda: dict(primal=list(), dual=list()))
        prop_dual = Program.propositional_dual(self.canonical_propositional_rules)
        pred_dual = Program.predicate_dual(self.canonical_predicate_rules)
        prop, pred = self.program_dicts
        for signature, rules in prop.items():
            signature_rules[signature]['primal'].extend(rules)
        for signature, rules in pred.items():
            signature_rules[signature]['primal'].extend(rules)
        for rule in prop_dual:
            head_signature = rule.head_signature
            signature_rules[head_signature]['primal' if rule.head.is_pos else 'dual'].append(rule)
        for rule in pred_dual:
            head_signature = rule.head_signature
            signature_rules[head_signature]['primal' if rule.head.is_pos else 'dual'].append(rule)
        return signature_rules

    @property
    def propositional_rules(self) -> Sequence[Rule]:
        return tuple(rule for rules in self.program_dicts[0].values() for rule in rules)

    @property
    def predicate_rules(self) -> Sequence[Rule]:
        return tuple(rule for rules in self.program_dicts[1].values() for rule in rules)

    @property
    def canonical_propositional_rules(self) -> Sequence[Rule]:
        return tuple(rule for rules in self.canonical_program_dicts[0].values() for rule in rules)

    @property
    def canonical_predicate_rules(self) -> Sequence[Rule]:
        return tuple(rule for rules in self.canonical_program_dicts[1].values() for rule in rules)

    def fmt(self, sep=' ', begin=None, end=None):
        b = begin + sep if begin is not None else ''
        e = sep + end if end is not None else ''
        return "{}{}{}".format(b, sep.join(map(str, self.rules)), e)

    def evaluate_backwards(self, *query: ClauseElement, nmr_check: bool = True):
        goal = Goal(*query)
        root = AndNode(subject=goal)
        program_dict = self.sASP_program_dict
        work = [root]
        while work:
            current = work.pop()
            if current.is_success:
                if current.is_root:
                    yield deepcopy(current)
                elif current.parent.is_success:
                    work.append(current.parent)
            elif not current.is_exhausted:
                current.expand(program_dict)
                work.extend(reversed(current.children))

    @staticmethod
    def propositional_dual(propositional_rules: Sequence[Rule]):
        dual_rules = []
        b2n: Dict[Sequence[ClauseElement], int] = {(): 0}
        n2b: Dict[int, Sequence[ClauseElement]] = {0: ()}
        n2h: Dict[int, Set[HeadClauseElement]] = {}
        h2n: Dict[HeadClauseElement, Set[int]] = {}
        ib: Set[Literal] = set()
        n = 0
        for rule in propositional_rules:
            assert isinstance(rule, NormalRule)
            head: BasicLiteral = rule.head
            body: Sequence[ClauseElement] = tuple(sorted(set(rule.body)))
            if body not in b2n:
                n += 1
                b2n[body] = n
                n2b[n] = body
            m = b2n[body]
            n2b[m] = body
            n2h.setdefault(m, set()).add(head)
            h2n.setdefault(head, set()).add(m)
        for h, ns in h2n.items():
            dual_head = -h
            dual_body = []
            dual_rule = None
            if ns and not any(n == 0 for n in ns):
                for n in ns:
                    b: Sequence[ClauseElement] = n2b[n]
                    if len(b) == 1:
                        if -b[0] not in dual_body:
                            dual_body.append(-b[0])
                        if isinstance(b[0], Literal):
                            ib.add(abs(b[0]))
                        dual_rule = NormalRule(dual_head, tuple(dual_body))
                    else:
                        assert len(b) > 1
                        if len(h2n[h]) == 1:
                            support_rule_head = -h
                            for l in b:
                                support_rule_body = (-l,)
                                support_rule = NormalRule(support_rule_head, support_rule_body)
                                dual_rules.append(support_rule)
                        else:
                            __b_n = BasicLiteral(atom=Atom(Function('__body', (Term(IntegerConstant(n)),))))
                            dual_body.append(-__b_n)
                            support_rule_head = -__b_n
                            for l in b:
                                if isinstance(l, Literal):
                                    ib.add(abs(l))
                                support_rule_body = (-l,)
                                support_rule = NormalRule(support_rule_head, support_rule_body)
                                dual_rules.append(support_rule)
                            dual_rule = NormalRule(dual_head, tuple(dual_body))

            if dual_rule is not None:
                dual_rules.append(dual_rule)
        for l in ib:
            if l not in h2n:
                dual_rules.append(NormalRule(-l))
        return dual_rules

    @staticmethod
    def predicate_dual(predicate_rules: Sequence[Rule]):
        dual_rules = []
        b2n: Dict[Sequence[ClauseElement], int] = {(): 0}
        n2b: Dict[int, Sequence[ClauseElement]] = {0: ()}
        n2h: Dict[int, Set[HeadClauseElement]] = {}
        h2n: Dict[HeadClauseElement, Set[int]] = {}
        ib: Set[ClauseElement] = set()
        n = 0
        for rule in predicate_rules:
            assert isinstance(rule, NormalRule)
            head: BasicLiteral = rule.head
            body: Sequence[ClauseElement] = tuple(sorted(set(rule.body)))
            if body not in b2n:
                n += 1
                b2n[body] = n
                n2b[n] = body
            m = b2n[body]
            n2b[m] = body
            n2h.setdefault(m, set()).add(head)
            h2n.setdefault(head, set()).add(m)
        for h, ns in h2n.items():
            dual_head = -h
            dual_body = []
            dual_rule = None
            head_variables = h.variables

            if ns and not any(n == 0 for n in ns):
                for n in ns:
                    b: Sequence[ClauseElement] = n2b[n]
                    body_variables = set(variable for element in b for variable in element.variables)
                    existentials = body_variables - head_variables
                    func__bf_n_ = Function(name="__body_fails_", arguments=(
                        Term(IntegerConstant(n)),
                        Function(arguments=tuple(head_variables)),
                        Function(arguments=tuple(existentials))
                    ))
                    __bf_n_ = BasicLiteral(atom=Atom(func__bf_n_))
                    dual_body_literal = func__bf_n_
                    for existential in existentials:
                        dual_body_literal = Directive.forall(existential, dual_body_literal)
                    if not existentials:
                        dual_body_literal = __bf_n_
                    dual_body.append(dual_body_literal)

                    support_rule_head = __bf_n_
                    support_rules = []
                    for e in b:
                        support_rules.append(NormalRule(support_rule_head, (-e,)))

                    dual_rules.extend(support_rules)

                dual_rules.append(NormalRule(dual_head, dual_body))

            if dual_rule is not None:
                dual_rules.append(dual_rule)
        for l in ib:
            if l not in h2n:
                dual_rules.append(NormalRule(-l))
        return dual_rules

A = Variable('A')
B = Variable('B')
X = Variable('X')
Y = Variable('Y')
t_A_A = BasicLiteral(atom=Atom(Function(name='t', arguments=(A, A))))
t_A_B = BasicLiteral(atom=Atom(Function(name='t', arguments=(A, B))))
q_0 = BasicLiteral(atom=Atom(Function(name='q', arguments=(Term.zero(),))))
_a = Function('a')
_b = Function('b')
_c = Function('c')
_q_abc = Function(name='q', arguments=(_a, _b, _c))
_r = Function(name='r')
_s = Function(name='s')
p_ = BasicLiteral(atom=Atom(Function(name='p', arguments=(_q_abc, _r, _s))))
p_1 = BasicLiteral(atom=Atom(Function(name='p', arguments=(Term.one(),))))
q_A = BasicLiteral(atom=Atom(Function(name='q', arguments=(A,))))
p_A = BasicLiteral(atom=Atom(Function(name='p', arguments=(A,))))

p = BasicLiteral(atom=Atom(Function(name='p')))
q_X = BasicLiteral(atom=Atom(Function(name='q', arguments=(X,))))
q_Y = BasicLiteral(atom=Atom(Function(name='q', arguments=(Y,))))
Y_ne_a = Comparison(Y, ComparisonOperator.NotEqual, _a)
Y_e_a = Comparison(Y, ComparisonOperator.Equal, _a)

r1 = NormalRule(head=t_A_A)
r2 = NormalRule(head=q_0)
r3 = NormalRule(head=p_)
r4 = NormalRule(head=q_A, body=(p_A,))
r5 = NormalRule(head=p_1)
r6 = NormalRule(head=t_A_B)
r7 = NormalRule(head=p, body=(-q_X,))
r8 = NormalRule(head=q_Y, body=(Y_e_a,))
r9 = NormalRule(head=q_Y, body=(Y_ne_a,))

print("#" * 80)
print("NormalRule.variable_normal_form():")
print("-" * 20)

print(r1)
print(r1.variable_normal_form())
print(r2)
print(r2.variable_normal_form())
print(r3)
print(r3.variable_normal_form())
print(r3)
print(r3.variable_normal_form())
print(r4)
print(r4.variable_normal_form())
print(r5)
print(r5.variable_normal_form())
print(r6)
print(r6.variable_normal_form())
print(r7)
print(r7.variable_normal_form())
print(r8)
print(r8.variable_normal_form())
print(r9)
print(r9.variable_normal_form())

print("#" * 80)
print("Program.canonical_program_dicts:")
print("-" * 20)

program1 = Program(rules=(r1, r2, r3))
prop, pred = program1.canonical_program_dicts

print("-" * 10)
print(program1.fmt('\n'))

print("-" * 10)

for s, rs in prop.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

print("-" * 10)

for s, rs in pred.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

program2 = Program(rules=(r1, r2, r3, r4))
prop, pred = program2.canonical_program_dicts

print("-" * 10)
print(program2.fmt('\n'))

print("-" * 10)

for s, rs in prop.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

print("-" * 10)

for s, rs in pred.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

program3 = Program(rules=(r1, r2, r3, r4, r5))
prop, pred = program3.canonical_program_dicts

print("-" * 10)
print(program3.fmt('\n'))

print("-" * 10)

for s, rs in prop.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

print("-" * 10)

for s, rs in pred.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

program4 = Program(rules=(r7, r8, r9))
prop, pred = program4.canonical_program_dicts

print("-" * 10)
print(program4.fmt('\n'))

print("-" * 10)

for s, rs in prop.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

print("-" * 10)

for s, rs in pred.items():
    print("{}:".format(s))
    for r in rs:
        print(r)

print("#" * 80)
print("Program.predicate_dual:")
print("-" * 20)

duals = Program.predicate_dual(program1.canonical_predicate_rules)

print('\n'.join(map(str, duals)))
print("-" * 10)

duals = Program.predicate_dual(program4.canonical_predicate_rules)

print('\n'.join(map(str, duals)))
print("-" * 10)

print("#" * 80)
print("Program.sASP_program_dict:")
print("-" * 20)
sASP = program4.sASP_program_dict

primal = []
dual = []
for signature in sASP:
    pos = sASP[signature]['primal']
    neg = sASP[signature]['dual']
    primal.extend(pos)
    dual.extend(neg)

print("% Primal:")
print('\n'.join(map(str, primal)))
print("% Dual:")
print('\n'.join(map(str, dual)))
print("-" * 10)

print("#" * 80)
print("Unify:")
print("-" * 20)

_src_chs = CoinductiveHypothesesSet()
_dst_chs = CoinductiveHypothesesSet()

_unifiable1 = _src_chs.unify(q_0, q_X, _dst_chs)
print("{}: ".format(_unifiable1), _dst_chs)

_src_chs = CoinductiveHypothesesSet()
_dst_chs = CoinductiveHypothesesSet()

_unifiable2 = _src_chs.unify(q_X, q_0, _dst_chs)

print("{}: ".format(_unifiable2), _dst_chs)

_src_chs = CoinductiveHypothesesSet()
_dst_chs = CoinductiveHypothesesSet()

_unifiable3 = _src_chs.unify(q_X, p_1, _dst_chs)

print("{}: ".format(_unifiable3), _dst_chs)

_src_chs = CoinductiveHypothesesSet({q_X: {X: dict(Positive={Term.one()}, Negative={Term.zero()})}})
_dst_chs = CoinductiveHypothesesSet()

_unifiable4 = _src_chs.unify(q_X, q_Y, _dst_chs)

print(_src_chs)
print("{}: ".format(_unifiable4), _dst_chs)
