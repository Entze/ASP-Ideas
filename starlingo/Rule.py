from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence, TypeVar, Union, Optional

import clingo
import clingo.ast

from starlingo.Atom import Atom, ComparisonOperator
from starlingo.Literal import Literal, ConditionalLiteral
from starlingo.Symbol import Variable, Term, Symbol
from starlingo.util import typecheck

ForwardExternal = TypeVar('ForwardExternal', bound='External')
ForwardExternalType = TypeVar('ForwardExternalType', bound='ExternalType')
ForwardRule = TypeVar('ForwardRule', bound='Rule')
ForwardFact = TypeVar('ForwardFact', bound='Fact')
ForwardNormalRule = TypeVar('ForwardNormalRule', bound='NormalRule')
ForwardConstraint = TypeVar('ForwardConstraint', bound='Constraint')
ForwardChoiceRule = TypeVar('ForwardChoiceRule', bound='ChoiceRule')
ForwardDisjunctiveRule = TypeVar('ForwardDisjunctiveRule', bound='DisjunctiveRule')


class RuleLike:
    def is_rule(self) -> bool:
        return isinstance(self, Rule)

    def is_external(self) -> bool:
        return isinstance(self, External)

    @staticmethod
    def body_str(body):
        if any(isinstance(literal, ConditionalLiteral) for literal in body):
            return '; '.join(map(str, body))
        return ", ".join(map(str, body))


class Rule(RuleLike):

    def is_fact(self):
        if isinstance(self, Fact):
            return True
        elif self.is_normal_rule():
            assert isinstance(self, NormalRule) or isinstance(self, DisjunctiveRule)
            return not self.body
        else:
            return False

    def as_fact(self):
        if isinstance(self.head, Sequence):
            head = self.head[0]
        else:
            head = self.head
        return Fact(head)

    def is_normal_rule(self):
        if isinstance(self, NormalRule):
            return True
        if self.is_fact():
            return True
        elif isinstance(self, DisjunctiveRule):
            return len(self.head) == 1
        return False

    def is_disjunctive_rule(self):
        return isinstance(self, DisjunctiveRule)

    def is_constraint(self):
        if isinstance(self, Constraint):
            return True
        return False

    @classmethod
    def from_ast(cls, rule: clingo.ast.AST) -> ForwardRule:
        typecheck(rule, clingo.ast.ASTType.Rule, 'ast_type')
        if rule.head.ast_type is clingo.ast.ASTType.Literal:
            if rule.head.atom == clingo.ast.BooleanConstant(False):
                return Constraint.from_ast(rule)
            elif not rule.body:
                return Fact.from_ast(rule)
            else:
                return NormalRule.from_ast(rule)
        elif rule.head.ast_type is clingo.ast.ASTType.Aggregate:
            return ChoiceRule.from_ast(rule)
        elif rule.head.ast_type is clingo.ast.ASTType.Disjunction:
            return DisjunctiveRule.from_ast(rule)
        else:
            assert False, "Unknown Rule Type {}.".format(rule)


@dataclass(frozen=True, order=True)
class Fact(Rule):
    head: Literal = field(default_factory=Literal)

    @property
    def body(self) -> Sequence[Literal]:
        return ()

    def __str__(self) -> str:
        return "{}.".format(self.head)

    @classmethod
    def from_ast(cls, fact: clingo.ast.AST) -> ForwardFact:
        typecheck(fact, clingo.ast.ASTType.Rule, 'ast_type')
        assert not fact.body, "clingo.ast.AST {} should not have a body.".format(fact)
        typecheck(fact.head, clingo.ast.ASTType.Literal, 'ast_type')
        head = Literal.from_ast(fact.head)
        return Fact(head)


@dataclass(frozen=True, order=True)
class NormalRule(Rule):
    head: Literal = field(default_factory=Literal)
    body: Sequence[Literal] = ()

    def __str__(self) -> str:
        return "{} :- {}.".format(self.head, Rule.body_str(self.body))

    @classmethod
    def from_ast(cls, normal: clingo.ast.AST) -> ForwardNormalRule:
        typecheck(normal.head, clingo.ast.ASTType.Literal, 'ast_type')
        assert normal.body, "clingo.ast.AST {} should have a non-empty body.".format(normal)
        head = Literal.from_ast(normal.head)
        body = tuple(Literal.from_ast(literal) for literal in normal.body)
        return NormalRule(head, body)


@dataclass(frozen=True, order=True)
class Constraint(Rule):
    body: Sequence[Literal] = ()

    def __str__(self) -> str:
        if self.body:
            return ":- {}.".format(Rule.body_str(self.body))
        else:
            return ":-."

    @classmethod
    def from_ast(cls, constraint: clingo.ast.AST) -> ForwardConstraint:
        assert constraint.head.atom == clingo.ast.BooleanConstant(
            False), "clingo.ast.AST {} should have head {}, but has {}.".format(constraint,
                                                                                clingo.ast.BooleanConstant(False),
                                                                                constraint.head.atom)
        body = tuple(Literal.from_ast(literal) for literal in constraint.body)
        return Constraint(body)


@dataclass(frozen=True, order=True)
class Guard:
    comparison: ComparisonOperator = field(default=ComparisonOperator.LessEqual)
    term: Union[Variable, Term] = field(default_factory=Term)


@dataclass(frozen=True, order=True)
class ChoiceRule(Rule):
    left_guard: Union[Guard, None] = None
    head: Sequence[ConditionalLiteral] = ()
    right_guard: Union[Guard, None] = None
    body: Sequence[Literal] = ()

    def __str__(self) -> str:
        if self.left_guard is None:
            left_guard_str = ''
        else:
            left_guard_str = "{} {} ".format(self.left_guard.term, self.left_guard.comparison)
        if self.right_guard is None:
            right_guard_str = ''
        else:
            right_guard_str = " {} {}".format(self.right_guard.comparison, self.right_guard.term)
        if self.body:
            body_str = " :- {}.".format(Rule.body_str(self.body))
        else:
            body_str = '.'
        return "{}{}{}{}{}{}".format(left_guard_str, '{', '; '.join(map(str, self.head)), '}', right_guard_str,
                                     body_str)

    @staticmethod
    def _get_guard(guard: Optional[clingo.ast.AST]) -> Guard:
        if guard is not None:
            typecheck(guard, clingo.ast.ASTType.AggregateGuard, 'ast_type')
            term = guard.term
            if term.ast_type is clingo.ast.ASTType.SymbolicTerm:
                term = Symbol.from_ast(term)
            elif term.ast_type is clingo.ast.ASTType.Variable:
                term = Variable.from_ast(term)
            else:
                assert False, "Unknown clingo.ast.ASTType {} of clingo.ast.AST {}.".format(term.ast_type, term)
            guard = Guard(ComparisonOperator(guard.comparison), term)
        return guard

    @classmethod
    def from_ast(cls, choice: clingo.ast.AST) -> ForwardChoiceRule:
        typecheck(choice.head, clingo.ast.ASTType.Aggregate, 'ast_type')
        elements = tuple(Literal.from_ast(element) for element in choice.head.elements)

        left_guard = ChoiceRule._get_guard(choice.head.left_guard)
        right_guard = ChoiceRule._get_guard(choice.head.right_guard)
        body = tuple(Literal.from_ast(literal) for literal in choice.body)
        return ChoiceRule(left_guard, elements, right_guard, body)


@dataclass(frozen=True, order=True)
class DisjunctiveRule(Rule):
    head: Sequence[ConditionalLiteral] = ()
    body: Sequence[Literal] = ()

    def __str__(self):
        if self.body:
            body_str = " :- {}.".format(Rule.body_str(self.body))
        else:
            body_str = '.'
        return "{}{}".format(' | '.join(map(str, self.head)), body_str)

    @classmethod
    def from_ast(cls, disjunctive_rule: clingo.ast.AST) -> ForwardDisjunctiveRule:
        typecheck(disjunctive_rule.head, clingo.ast.ASTType.Disjunction, 'ast_type')
        elements = tuple(ConditionalLiteral.from_ast(cond_literal) for cond_literal in disjunctive_rule.head.elements)
        body = tuple(Literal.from_ast(literal) for literal in disjunctive_rule.body)
        return DisjunctiveRule(elements, body)


class ExternalType(IntEnum):
    False_ = 0
    True_ = 1
    Free = 2
    Release = 3

    @staticmethod
    def from_truth_value(tv: clingo.TruthValue) -> ForwardExternalType:
        if tv is clingo.TruthValue.False_:
            return ExternalType.False_
        elif tv is clingo.TruthValue.True_:
            return ExternalType.True_
        elif tv is clingo.TruthValue.Free:
            return ExternalType.Free
        else:
            assert tv is clingo.TruthValue.Release
            return ExternalType.Release


@dataclass(frozen=True, order=True)
class External(RuleLike):
    atom: Atom = field(default_factory=Atom)
    body: Sequence[Literal] = ()
    external_type: ExternalType = ExternalType.False_

    def __str__(self):
        if not self.body:
            return "#external {}. [{}]".format(self.atom, self.external_type.name)
        return "#external {} : {}. [{}]".format(self.atom, ', '.join(map(str, self.body)), self.external_type.name)

    @classmethod
    def from_ast(cls, external: clingo.ast.AST) -> ForwardExternal:
        typecheck(external, clingo.ast.ASTType.External, 'ast_type')
        atom = Atom.from_ast(external.atom)
        body = tuple(Literal.from_ast(literal) for literal in external.body)
        external_type = ExternalType.from_truth_value(external.external_type)
        return External(atom, body, external_type)
