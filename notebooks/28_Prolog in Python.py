import copy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Union, Sequence, Optional, Dict, TypeVar


class TermType(IntEnum):
    FUNCTION = 0
    VARIABLE = 1
    INTEGER = 2


@dataclass(frozen=True)
class Variable:
    name: str

    def __str__(self):
        return self.name


ForwardTerm = TypeVar('ForwardTerm', bound='Term')


@dataclass(frozen=True)
class Term:
    termType: TermType = TermType.FUNCTION
    symbol: Union[str, int, Variable, None] = None
    arguments: Sequence[ForwardTerm] = ()

    def __str__(self) -> str:
        if self.termType is TermType.FUNCTION:
            return self.__str_function()
        elif self.termType is TermType.VARIABLE:
            return self.__str_variable()
        elif self.termType is TermType.INTEGER:
            return self.__str_integer()
        else:
            assert False, "Unknown TermType '{}'.".format(self.termType.name)

    def __str_function(self) -> str:
        if self.symbol is None and not self.arguments:
            return "()"
        elif self.symbol is None:
            return "({})".format(','.join(map(str, self.arguments)))
        elif not self.arguments:
            return str(self.symbol)
        else:
            return "{}({})".format(self.symbol, ','.join(map(str, self.arguments)))

    def __str_variable(self) -> str:
        return str(self.symbol)

    def __str_integer(self) -> str:
        return str(self.symbol)

    @staticmethod
    def new_variable(name: str) -> ForwardTerm:
        return Term(termType=TermType.VARIABLE, symbol=Variable(name))

    @staticmethod
    def new_integer(num: int) -> ForwardTerm:
        return Term(termType=TermType.INTEGER, symbol=num)

    @staticmethod
    def new_function(name: str, arguments: Sequence[ForwardTerm] = ()) -> ForwardTerm:
        return Term(termType=TermType.FUNCTION, symbol=name, arguments=arguments)

    @staticmethod
    def new_constant(name: str) -> ForwardTerm:
        return Term.new_function(name)


@dataclass
class Rule:
    head: Optional[Term] = None
    body: Sequence[Term] = ()

    def __str__(self) -> str:
        if self.head is None and not self.body:
            return ":-."
        elif self.head is None:
            return ":- {}.".format(', '.join(map(str, self.body)))
        elif not self.body:
            return "{}.".format(self.head)
        else:
            return "{} :- {}.".format(self.head, ', '.join(map(str, self.body)))


ForwardGoal = TypeVar('ForwardGoal', bound='Goal')


@dataclass
class Goal:
    goal: Optional[Rule] = None
    parent: Optional[ForwardGoal] = None
    children : Sequence[ForwardGoal] = field(default_factory=list)
    env: Dict[Variable, Term] = field(default_factory=dict)
    inx: int = 0


def unify(src_term: Term, src_env: Dict[Variable, Term], dest_term: Optional[Term], dest_env: Dict[Variable, Term]):
    if dest_term is None:
        return False
    if src_term.termType is not TermType.FUNCTION:
        return False
    if dest_term.termType is not TermType.FUNCTION:
        return False
    if src_term.symbol != dest_term.symbol:
        return False
    nargs = len(src_term.arguments)
    if nargs != len(dest_term.arguments):
        return False
    for i in range(nargs):
        src_arg: Term = src_term.arguments[i]
        dest_arg: Term = dest_term.arguments[i]
        if src_arg.termType is TermType.VARIABLE:
            src_val = src_env.get(src_arg.symbol)
        else:
            src_val = src_arg
        if src_val is not None:
            if dest_arg.termType is TermType.VARIABLE:
                dest_val = dest_env.get(dest_arg.symbol)
                if dest_val is None:
                    dest_env[dest_arg.symbol] = src_val
                elif dest_val != src_val:
                    return False
            elif dest_arg != src_val:
                return False
    return True


def search(term: Term, rules: Sequence[Rule] = ()):

    root = Goal(goal=Rule(head=Term(), body=(term,)))
    goal_envs = []
    proof_trees = []
    stack = [root]
    while stack:
        current = stack.pop()
        if current.inx >= len(current.goal.body):
            if current.parent is None:
                if current.env:
                    print(current.env)
                else:
                    print("Yes")
                goal_envs.append(current.env)
                proof_trees.append(current)
            else:
                parent = copy.deepcopy(current.parent)
                unify(current.goal.head, current.env, parent.rule.body[parent.inx], parent.env)
                parent.inx += 1
                stack.append(parent)
        else:
            term = current.goal.body[current.inx]
            for rule in rules:
                child_env = {}
                unifiable = unify(term, current.env, rule.head, child_env)
                if unifiable:
                    child = Goal(env=child_env, parent=current, goal=rule)
                    current.children.append(child)
                    stack.append(child)
    if not goal_envs:
        print("No")
    return goal_envs,proof_trees


X = Term.new_variable('X')
Y = Term.new_variable('Y')
A = Term.new_variable('A')
bill = Term.new_constant('bill')
frank = Term.new_constant('frank')
alice = Term.new_constant('alice')
alex = Term.new_constant('alex')

program = [
    Rule(head=Term.new_function('child', (X, Y)), body=(Term.new_function('mother', (Y, X)),)),
    Rule(head=Term.new_function('child', (X, Y)), body=(Term.new_function('father', (Y, X)),)),

    Rule(head=Term.new_function('son', (X, Y)),
         body=(Term.new_function('child', (X, Y)), Term.new_function('boy', (X,)))),

    Rule(head=Term.new_function('boy', (bill,))),
    Rule(head=Term.new_function('boy', (frank,))),
    Rule(head=Term.new_function('mother', (alice, bill))),
    Rule(head=Term.new_function('father', (alex, bill)))
]
print('\n'.join(map(str, program)))

search_term = Term.new_function('son', (bill, A))
print(f"{search_term}?")

search(search_term, program)
