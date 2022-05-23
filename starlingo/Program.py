from dataclasses import dataclass, field
from typing import Sequence, MutableSequence, Optional, Iterator

import clingo.ast

from starlingo.Atom import Atom
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
        self.parameters = tuple(Function(parameter.name) for parameter in parameters)
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


def from_string(program: str) -> Sequence[Program]:
    t = _FromASTTransformer()
    clingo.ast.parse_string(program, t.visit)
    t.flush()
    return t.programs


def evaluate_forwards(programs: Sequence[Program],
                      ctl: Optional[clingo.Control] = None,
                      parts=(('base', ()),),
                      report=False,
                      report_models=True,
                      report_result=True) -> Iterator[Sequence[Atom]]:
    if ctl is None:
        ctl = clingo.Control()
        ctl.configuration.solve.models = 0
    ctl.add('base', [], '\n\n'.join(map(str, programs)))
    ctl.ground(parts)
    with ctl.solve(yield_=True) as solve_handle:
        models = 0
        for model in solve_handle:
            symbols = sorted(model.symbols(shown=True))
            if report and report_models:
                print("Answer {}:".format(model.number), end=' ')
                print("{",
                      '\n'.join(map(str, symbols)), "}", sep='\n')
            atoms = tuple(Atom.from_clingo_symbol(symbol) for symbol in symbols)
            models += 1
            yield atoms
        if report and report_result:
            solve_result = solve_handle.get()
            print(solve_result, end='')
            if solve_result.satisfiable:
                print(" {}{}".format(models, '' if solve_result.exhausted else '+'))
