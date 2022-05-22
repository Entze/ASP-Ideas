from dataclasses import dataclass, field
from typing import Sequence, MutableSequence, Optional

import clingo.ast

from starlingo.Rule import RuleLike, Rule, External
from starlingo.Symbol import Function
from starlingo.util import open_join_close


@dataclass(frozen=True, order=True)
class Program:
    name: str = field(default='base')
    parameters: Sequence[Function] = field(default_factory=tuple)
    rules: Sequence[RuleLike] = field(default_factory=tuple)

    def custom_str(self, sep=' ', start='', end='') -> str:
        strs = (
            '#program {}{}.'.format(self.name, open_join_close(',', '(', ')', map(str, self.parameters))),
            *(map(str, self.rules)))
        return "{}{}{}".format(start, sep.join(strs), end)

    def __str__(self):
        return self.custom_str()


class _FromASTTransformer(clingo.ast.Transformer):

    def __init__(self):
        self.name: Optional[str] = None
        self.parameters: Optional[Sequence[Function]] = None
        self.program_rules: Optional[MutableSequence[RuleLike]] = None
        self.programs: MutableSequence[Program] = []

    def flush(self, name: Optional[str] = None, parameters: Sequence[clingo.ast.AST] = ()):
        if self.name is not None:
            self.programs.append(Program(self.name, self.parameters, self.program_rules))
        self.name = name
        self.parameters = tuple(Function.from_ast(parameter) for parameter in parameters)
        self.program_rules = []

    def visit_Program(self, program: clingo.ast.AST) -> clingo.ast.AST:
        self.flush(program.name, program.parameters)
        return program

    def visit_Rule(self, rule: clingo.ast.AST) -> clingo.ast.AST:
        if self.program_rules is not None:
            self.program_rules.append(Rule.from_ast(rule))
        return rule

    def visit_External(self, external: clingo.ast.AST) -> clingo.ast.AST:
        if self.program_rules is not None:
            self.program_rules.append(External.from_ast(external))
        return external


def from_ast(program: str) -> Sequence[Program]:
    t = _FromASTTransformer()
    clingo.ast.parse_string(program, t.visit)
    t.flush()
    return t.programs
