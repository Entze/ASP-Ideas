from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Optional, Sequence, Iterator, Dict, Set, TypeVar, MutableSequence


# %%
class Symbol:
    pass


# %%
class TopLevelSymbol(Symbol):
    pass


# %%
@dataclass(order=True, frozen=True)
class Function(TopLevelSymbol):
    name: Optional[str] = None

    def __str__(self):
        if self.name is None:
            return '()'
        else:
            return self.name


# %%
@dataclass(order=True, frozen=True)
class Atom:
    symbol: TopLevelSymbol = field(default_factory=Function)

    def __str__(self):
        return str(self.symbol)


# %%
class Literal:
    pass


# %%
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


# %%
@dataclass(order=True, frozen=True)
class BasicLiteral(Literal):
    sign: Sign = Sign.NoSign
    atom: Atom = field(default_factory=Atom)

    def __str__(self):
        if self.sign is Sign.NoSign:
            return "{}".format(self.atom)
        else:
            return "{} {}".format(self.sign, self.atom)

    def __neg__(self):
        return BasicLiteral(Sign((self.sign ^ 1) % 2), self.atom)

    def __abs__(self):
        return BasicLiteral(Sign.NoSign, self.atom)

    @property
    def is_pos(self) -> bool:
        return self.sign is Sign.NoSign

    @property
    def is_neg(self) -> bool:
        return self.sign is Sign.Negation


# %%
class Rule:
    @staticmethod
    def fmt_body(body: Sequence[BasicLiteral]):
        return ', '.join(map(str, body))


# %%
@dataclass(order=True, frozen=True)
class NormalRule(Rule):
    head: BasicLiteral = field(default_factory=BasicLiteral)
    body: Sequence[BasicLiteral] = ()

    def __str__(self):
        if self.body:
            return "{} :- {}.".format(self.head, Rule.fmt_body(self.body))
        else:
            return "{}.".format(self.head)


# %%
@dataclass(order=True, frozen=True)
class IntegrityConstraint(Rule):
    body: Sequence[BasicLiteral] = ()

    @property
    def head(self):
        return False

    def __str__(self):
        if self.body:
            return '#false :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#false.'


# %%
@dataclass(order=True, frozen=True)
class Goal(Rule):
    body: Sequence[BasicLiteral] = ()

    @property
    def head(self):
        return True

    def __str__(self):
        if self.body:
            return '#true :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#true.'


# %%
ForwardProof = TypeVar('ForwardProof', bound='Proof')


@dataclass
class Proof:
    parent: ForwardProof = field(repr=False, default=None)
    idx: int = 0
    subject: Optional[Rule] = field(default=None)
    children: MutableSequence[ForwardProof] = field(repr=False, default_factory=list)
    hypotheses: Set[Literal] = field(default_factory=set)


# %%
@dataclass(order=True, frozen=True)
class Program:
    rules: Sequence[Rule] = ()

    def fmt(self, sep=' ', begin=None, end=None):
        b = begin + sep if begin is not None else ''
        e = sep + end if end is not None else ''
        return "{}{}{}".format(b, sep.join(map(str, self.rules)), e)

    def __str__(self):
        return self.fmt()

    @cached_property
    def dual(self):  # type Program
        return Program.dual_of(self.rules)

    @cached_property
    def sASP(self):
        sasp_rules = list(self.rules)
        sasp_rules.extend(self.dual_of(tuple(self.non_constraint_rules)).rules)

        chk_rules = []
        nmr_chk_head = BasicLiteral(atom=Atom(Function("__nmr_chk")))
        nmr_chk_body = []
        for i, c_rule in enumerate(self.constraint_rules):
            chk_head = BasicLiteral(atom=Atom(Function("__chk_{}_{}".format(c_rule.head.atom.symbol.name, i))))
            chk_body = (-c_rule.head, *(body_literal for body_literal in c_rule.body if -c_rule.head != body_literal))
            chk_rule = NormalRule(head=chk_head, body=chk_body)
            nmr_chk_body.append(-chk_head)

            chk_rules.append(chk_rule)
        sasp_rules.extend(chk_rules)
        nmr_chk_rule = NormalRule(nmr_chk_head, nmr_chk_body)
        sasp_rules.append(nmr_chk_rule)
        return Program(rules=sasp_rules)

    @cached_property
    def reachable(self) -> Dict[Rule, Set[Literal]]:
        reachable = defaultdict(set)
        for rule in self.rules:
            considered = set()
            literal_stack = []
            naf_stack = []
            literal_stack.extend(rule.body)
            naf_stack.extend([0 for _ in rule.body])
            while literal_stack:
                literal = literal_stack.pop()
                naf = naf_stack.pop()
                naf = (naf + literal.is_neg) % 2
                reachable[rule].add(BasicLiteral(Sign(naf), literal.atom))
                for adj in self.rules:
                    if adj not in considered and adj.head == abs(literal):
                        considered.add(adj)
                        literal_stack.extend(adj.body)
                        naf_stack.extend([naf for _ in adj.body])
        return reachable

    @property
    def constraint_rules(self) -> Iterator[Rule]:
        for rule, reachable in self.reachable.items():
            if -rule.head in reachable:
                yield rule

    @property
    def non_constraint_rules(self) -> Iterator[Rule]:
        for rule, reachable in self.reachable.items():
            if rule.head in reachable or -rule.head not in reachable:
                yield rule

    def evaluate_top_down(self, *literals: Literal):
        hypothesis_set = set()
        for rule in self.rules:
            if not rule.body:
                hypothesis_set.add(rule.head)
        proofs = []
        rules = self.sASP.rules
        __nmr_chk = BasicLiteral(atom=Atom(Function('__nmr_chk')))
        root = Proof(subject=Goal(body=(*literals, __nmr_chk)), hypotheses=hypothesis_set)
        stack = [root]
        while stack:
            current = stack.pop()
            assert isinstance(current, Proof)
            if set(current.subject.body) <= current.hypotheses:
                if current.parent is None:
                    proofs.append(current)
                else:
                    parent = deepcopy(current.parent)
                    parent.hypotheses = deepcopy(current.hypotheses)
                    stack.append(parent)
            else:
                literal = None
                for body_literal in current.subject.body:
                    if body_literal not in current.hypotheses:
                        literal = body_literal
                        break
                if literal is not None:
                    for rule in rules:
                        if rule.head == literal:
                            if not any(-body_literal in current.hypotheses for body_literal in rule.body):
                                hypotheses = deepcopy(current.hypotheses)
                                hypotheses.add(literal)
                                child = Proof(parent=current, subject=rule, hypotheses=hypotheses)
                                current.children.append(child)
                                stack.append(child)

        return proofs

    @staticmethod
    def dual_of(rules):
        lit_rules = dict()
        for rule in rules:
            head = rule.head
            if isinstance(rule, NormalRule):
                lit_rules.setdefault(head, set()).add(rule)

        dual_rules = []
        for literal, rules in lit_rules.items():
            if isinstance(literal, bool):
                pass
            elif isinstance(literal, BasicLiteral):
                if len(rules) == 1:
                    rules = tuple(rules)
                    rule = rules[0]
                    dual_head = -literal
                    if not rule.body:
                        dual_bodies = ((),)
                    else:
                        dual_bodies = []
                        for body_literal in rule.body:
                            dual_bodies.append((-body_literal,))
                    for dual_body in dual_bodies:
                        dual_rules.append(NormalRule(head=dual_head, body=dual_body))
                elif len(rules) > 1:
                    dual_head = -literal
                    dual_body = []
                    support_dual_rules = []
                    for i, rule in enumerate(rules):
                        if len(rule.body) == 1:
                            dual_body.append(-rule.body[0])
                        elif len(rule.body) > 1:
                            support_dual_head = BasicLiteral(
                                atom=Atom(Function("__not_{}_{}".format(literal.atom.symbol.name, i))))
                            dual_body.append(support_dual_head)
                            if not rule.body:
                                support_dual_bodies = ((),)
                            else:
                                support_dual_bodies = []
                                for body_literal in rule.body:
                                    support_dual_bodies.append((-body_literal,))
                            for support_dual_body in support_dual_bodies:
                                support_dual_rules.append(NormalRule(head=support_dual_head, body=support_dual_body))
                    dual_rules.append(NormalRule(head=dual_head, body=dual_body))
                    dual_rules.extend(support_dual_rules)
        return Program(rules=dual_rules)


# %%
q = BasicLiteral(atom=Atom(Function('q')))
p = BasicLiteral(atom=Atom(Function('p')))
r = BasicLiteral(atom=Atom(Function('r')))
###
a = BasicLiteral(atom=Atom(Function('a')))
b = BasicLiteral(atom=Atom(Function('b')))
c = BasicLiteral(atom=Atom(Function('c')))
d = BasicLiteral(atom=Atom(Function('d')))
e = BasicLiteral(atom=Atom(Function('e')))
f = BasicLiteral(atom=Atom(Function('f')))
k = BasicLiteral(atom=Atom(Function('k')))

p1 = Program(rules=(
    NormalRule(head=p, body=(-q,)),
    NormalRule(head=q, body=(-r,)),
    NormalRule(head=r, body=(-p,)),
    NormalRule(head=q, body=(-p,)),
))
print(p1.fmt('\n'))  # AS: {{q, r}}
print('-' * 10)
d1 = p1.dual
print(d1.fmt('\n'))
print('-' * 10)
s1 = p1.sASP
print(s1.fmt('\n'))

# %%
answer_sets = p1.evaluate_top_down(q)
print("q:")
for answer_set in answer_sets:
    print("{", end='')
    print(', '.join(map(str, answer_set.hypotheses)), end='')
    print("}")

print("r:")
answer_sets = p1.evaluate_top_down(r)
for answer_set in answer_sets:
    print("{", end='')
    print(', '.join(map(str, answer_set.hypotheses)), end='')
    print("}")
