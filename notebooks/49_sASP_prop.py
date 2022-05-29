from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Optional, Sequence, Iterator, Dict, Set, TypeVar, Union, MutableSequence


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

    @property
    def head(self) -> BasicLiteral:
        if hasattr(self, '_head'):
            return getattr(self, '_head')
        raise NotImplementedError

    @property
    def body(self) -> Sequence[BasicLiteral]:
        if hasattr(self, '_body'):
            return getattr(self, '_body')
        raise NotImplementedError

    @staticmethod
    def fmt_body(body: Sequence[BasicLiteral]):
        return ', '.join(map(str, body))


# %%
@dataclass(order=True, frozen=True)
class NormalRule(Rule):
    _head: BasicLiteral = field(default_factory=BasicLiteral)
    _body: Sequence[BasicLiteral] = ()

    @property
    def head(self) -> BasicLiteral:
        return self._head

    @property
    def body(self) -> Sequence[BasicLiteral]:
        return self._body

    def __str__(self):
        if self.body:
            return "{} :- {}.".format(self.head, Rule.fmt_body(self.body))
        else:
            return "{}.".format(self.head)


# %%
@dataclass(order=True, frozen=True)
class IntegrityConstraint(Rule):
    _body: Sequence[BasicLiteral] = ()

    @property
    def head(self) -> bool:
        return False

    @property
    def body(self) -> Sequence[BasicLiteral]:
        return self._body

    def __str__(self):
        if self.body:
            return '#false :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#false.'


# %%
@dataclass(order=True, frozen=True)
class Goal(Rule):
    _body: Sequence[BasicLiteral] = ()

    @property
    def head(self):
        return True

    @property
    def body(self) -> Sequence[BasicLiteral]:
        return self._body

    def __str__(self):
        if self.body:
            return '#true :- {}.'.format(Rule.fmt_body(self.body))
        else:
            return '#true.'


# %%
_ForwardBaseNode = TypeVar('_ForwardBaseNode', bound='_BaseNode')


@dataclass
class _BaseNode:
    subject: Union[Literal, Rule, None]
    hypotheses: Set[Literal]
    parent: Optional[_ForwardBaseNode]
    children: MutableSequence[_ForwardBaseNode]
    index: int

    @property
    def is_expanded(self):
        return self.children is not None

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_leaf(self):
        return self.is_expanded and not self.children

    @property
    def is_complete(self) -> bool:
        return False

    def is_exhausted(self, rules: Sequence[Rule]) -> bool:
        raise NotImplementedError

    def expand_all(self, rules: Sequence[Rule]) -> Sequence[_ForwardBaseNode]:
        raise NotImplementedError

    def expand(self, rules: Sequence[Rule]) -> Optional[_ForwardBaseNode]:
        raise NotImplementedError

    def propagate_parent(self) -> _ForwardBaseNode:
        return self.parent


@dataclass
class Node(_BaseNode):
    subject: Union[Literal, Rule, None] = field(default=None)
    hypotheses: Set[Literal] = field(default_factory=set)
    parent: Optional[_BaseNode] = field(default=None)
    children: Optional[MutableSequence[_BaseNode]] = field(default=None)
    index: int = field(default=0)


ForwardAndNode = TypeVar('ForwardAndNode', bound='AndNode')


@dataclass
class OrNode(Node):
    subject: Optional[Literal] = field(default=None)
    children: Optional[MutableSequence[ForwardAndNode]] = field(default=None)

    @property
    def is_complete(self) -> bool:
        return self.subject in self.hypotheses

    def is_exhausted(self, rules: Sequence[Rule]) -> bool:
        return self.index >= len(rules)

    def expand_all(self, rules: Sequence[Rule]) -> Sequence[ForwardAndNode]:
        self.children = []
        for rule in rules:
            if rule.head != self.subject:
                continue
            if any(-body_literal in self.hypotheses for body_literal in rule.body):
                continue
            hypotheses = deepcopy(self.hypotheses)
            hypotheses.add(self.subject)
            child = AndNode(subject=rule, hypotheses=hypotheses, parent=self)
            self.children.append(child)
        return self.children

    def expand(self, rules: Sequence[Rule]) -> Optional[ForwardAndNode]:
        if not self.is_expanded:
            self.children = []
        elif self.is_exhausted(rules):
            return None
        if -self.subject in self.hypotheses:
            return None
        rule = rules[self.index]
        if rule.head != self.subject:
            return None
        if any(-body_literal in self.hypotheses for body_literal in rule.body):
            return None
        hypotheses = deepcopy(self.hypotheses)
        hypotheses.add(self.subject)
        child = AndNode(subject=rule, hypotheses=hypotheses, parent=self)
        self.children.append(child)
        return child

    def propagate_parent(self) -> ForwardAndNode:
        hypotheses = deepcopy(self.hypotheses)
        parent = AndNode(subject=self.parent.subject,
                         hypotheses=hypotheses,
                         parent=self.parent.parent,
                         children=self.parent.children,
                         index=self.parent.index + 1)
        self.parent = parent
        return parent


@dataclass
class AndNode(Node):
    subject: Optional[Rule] = field(default=None)
    children: Optional[MutableSequence[OrNode]] = field(default=None)

    @property
    def is_complete(self) -> bool:
        return all(body_literal in self.hypotheses for body_literal in self.subject.body)

    def is_exhausted(self, rules: Sequence[Rule] = ()) -> bool:
        return self.index >= len(self.subject.body)

    def expand_all(self, rules: Sequence[Rule]) -> Sequence[OrNode]:
        self.children = []
        for body_literal in self.subject.body:
            if body_literal in self.hypotheses:
                continue
            hypotheses = deepcopy(self.hypotheses)
            child = OrNode(subject=body_literal, hypotheses=hypotheses, parent=self)
            self.children.append(child)
        return self.children

    def expand(self, rules: Sequence[Rule]) -> Optional[OrNode]:
        if not self.is_expanded:
            self.children = []
        if self.is_exhausted(rules):
            return None
        body_literal = self.subject.body[self.index]
        if body_literal in self.hypotheses:
            return None
        hypotheses = deepcopy(self.hypotheses)
        child = OrNode(subject=body_literal,
                       hypotheses=hypotheses,
                       parent=self)

        self.children.append(child)
        return child

    def propagate_parent(self) -> OrNode:
        hypotheses = deepcopy(self.hypotheses)
        parent = OrNode(subject=self.parent.subject,
                        hypotheses=hypotheses,
                        parent=self.parent.parent,
                        children=self.parent.children,
                        index=self.parent.index + 1)
        self.parent = parent
        return parent


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

        dual_facts = []
        considered = set()
        for rule in sasp_rules:
            for body_literal in rule.body:
                l = abs(body_literal)
                if l in considered:
                    continue
                if l not in self.reachable:
                    dual_facts.append(NormalRule(-abs(body_literal)))
                    considered.add(l)

        sasp_rules.extend(dual_facts)


        chk_rules = []
        nmr_chk_head = BasicLiteral(atom=Atom(Function("__nmr_chk")))
        nmr_chk_body = []
        for i, c_rule in enumerate(self.constraint_rules):
            chk_head = BasicLiteral(atom=Atom(Function("__chk_{}_{}".format(c_rule.head.atom.symbol.name, i))))
            chk_rule_ = NormalRule(chk_head, (c_rule.head,))
            chk_rules_ = (NormalRule(chk_head, (-body_literal,)) for body_literal in c_rule.body if
                          c_rule.body != -c_rule.head)
            chk_rules.append(chk_rule_)
            chk_rules.extend(chk_rules_)
            nmr_chk_body.append(chk_head)

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

    def evaluate_top_down(self, *literals: Literal) -> Sequence[Node]:
        __nmr_chk = BasicLiteral(atom=Atom(Function("__nmr_chk")))
        goal = Goal((*literals, __nmr_chk))
        root = AndNode(subject=goal)
        rules = self.sASP.rules
        proofs = []
        derivation_stack = [root]
        while derivation_stack:
            current = derivation_stack.pop()
            if current.is_complete:
                if current.is_root:
                    proofs.append(current)
                else:
                    new_parent = current.propagate_parent()
                    derivation_stack.append(new_parent)
            elif not current.is_exhausted(rules):
                child = current.expand(rules)
                if child is None:
                    current.index += 1
                    derivation_stack.append(current)
                else:
                    if isinstance(current, OrNode):
                        current.index += 1
                        derivation_stack.append(current)
                    derivation_stack.append(child)

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
                        dual_rules.append(NormalRule(dual_head, dual_body))
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
                                support_dual_rules.append(NormalRule(support_dual_head, support_dual_body))
                    dual_rules.append(NormalRule(dual_head, dual_body))
                    dual_rules.extend(support_dual_rules)
        return Program(rules=dual_rules)


# %% md

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


def solve(p: Program, *literals: Literal):
    proofs = p.evaluate_top_down(*literals)
    if not proofs:
        print("UNSAT")
    else:
        for i, proof in enumerate(proofs):
            print("Answer {}:".format(i), end=' ')
            print("{", end=' ')
            print(' '.join(map(str, proof.hypotheses)), end='')
            if not proof.hypotheses:
                print(' ', end='')
            print("}")
        print("SAT {}+".format(len(proofs)))


# %%
p1 = Program(rules=(
    NormalRule(p, (-q,)),
    NormalRule(q, (-r,)),
    NormalRule(r, (-p,)),
    NormalRule(q, (-p,)),
))
print(p1.fmt('\n'))  # AS: {{q, r}}
print('-' * 10)
d1 = p1.dual
print(d1.fmt('\n'))
print('-' * 10)
s1 = p1.sASP
print(s1.fmt('\n'))

# %%
print('#' * 3, p, '#' * 3)
solve(p1, p)
print('#' * 3, q, '#' * 3)
solve(p1, q)
print('#' * 3, r, '#' * 3)
solve(p1, r)
print('#' * 3, q, r, '#' * 3)
solve(p1, q, r)

# %%
p2 = Program(rules=(
    NormalRule(q, (-r,)),
    NormalRule(r, (-q,)),
    NormalRule(p, (-p,)),
    NormalRule(p, (-r,)),
))
print(p2.fmt('\n'))  # AS: {{q, p}}
print('-' * 10)
d2 = p2.dual
print(d2.fmt('\n'))
print('-' * 10)
s2 = p2.sASP
print(s2.fmt('\n'))
# %%
print('#' * 3, p, '#' * 3)
solve(p2, p)
print('#' * 3, q, '#' * 3)
solve(p2, q)
print('#' * 3, r, '#' * 3)
solve(p2, r)
print('#' * 3, q, p, '#' * 3)
solve(p2, q, p)

# %%
p3 = Program(rules=(
    NormalRule(a, (b, d)),
    NormalRule(b, (d,)),
    NormalRule(c, (d,)),
    NormalRule(d, ()),
))
print(p3.fmt('\n'))
print('-' * 10)
d3 = p3.dual
print(d3.fmt('\n'))
print('-' * 10)
s3 = p3.sASP
print(s3.fmt('\n'))
print('#' * 3, '#' * 3)
solve(p3)
print('#' * 3, a, '#' * 3)
solve(p3, a)
print('#' * 3, b, '#' * 3)
solve(p3, b)
print('#' * 3, c, '#' * 3)
solve(p3, c)
print('#' * 3, d, '#' * 3)
solve(p3, d)
print('#' * 3, a, b, c, d, '#' * 3)
solve(p3, a, b, c, d)

# %%
p4 = Program(rules=(
    NormalRule(a, (k, -b)),
    NormalRule(k, (e, -b)),
    NormalRule(c, (a, b)),
    NormalRule(b, (-a,)),
    NormalRule(c, (k,)),
    NormalRule(f, (e, -k, -c)),
    NormalRule(e),
))
print(p4.fmt('\n'))
print('-' * 10)
d4 = p4.dual
print(d4.fmt('\n'))
print('-' * 10)
s4 = p4.sASP
print(s4.fmt('\n'))

print('#' * 3, '#' * 3)
solve(p4)
print('#' * 3, b, '#' * 3)
solve(p4, b)
print('#' * 3, e, '#' * 3)
solve(p4, e)
print('#' * 3, f, '#' * 3)
solve(p4, f)
print('#' * 3, a, '#' * 3)
solve(p4, a)
print('#' * 3, c, '#' * 3)
solve(p4, c)
print('#' * 3, k, '#' * 3)
solve(p4, k)
print('#' * 3, b, e, f, '#' * 3)
solve(p4, b, e, f)
print('#' * 3, a, c, '#' * 3)
solve(p4, a, c)
print('#' * 3, a, c, e, k, '#' * 3)
solve(p4, a, c, e, k)

# %%
p5 = Program(rules=(
    NormalRule(p, (a, -q)),
    NormalRule(q, (b, -r)),
    NormalRule(r, (c, -p)),
    NormalRule(q, (d, -p)),
))
print(p5.fmt('\n'))
print('-' * 10)
d5 = p5.dual
print(d5.fmt('\n'))
print('-' * 10)
s5 = p5.sASP
print(s5.fmt('\n'))
# %%
print('#' * 3, '#' * 3)
solve(p5)
print('#' * 3, p, '#' * 3)
solve(p5, p)
print('#' * 3, q, '#' * 3)
solve(p5, q)
print('#' * 3, r, '#' * 3)
solve(p5, r)
print('#' * 3, a, '#' * 3)
solve(p5,  a)
print('#' * 3, b, '#' * 3)
solve(p5,  b)
print('#' * 3, c, '#' * 3)
solve(p5, c)

# %%
p6 = Program(rules=(
    NormalRule(a, (-b,)),
    NormalRule(b, (-a,)),
))
print(p6.fmt('\n'))  # AS: {{q, p}}
print('-' * 10)
d6 = p6.dual
print(d6.fmt('\n'))
print('-' * 10)
s6 = p6.sASP
print(s6.fmt('\n'))
print('#' * 3, '#' * 3)
solve(p6)
print('#' * 3, a, '#' * 3)
solve(p6,  a)
print('#' * 3, b, '#' * 3)
solve(p6,  b)
print('#' * 3,a, b, '#' * 3)
solve(p6, a, b)

# %%
p7 = Program(rules=(
    NormalRule(a, (-b,)),
    NormalRule(b, (-c,)),
    NormalRule(c, (-a,)),
))
print(p7.fmt('\n'))  # AS: {{q, p}}
print('-' * 10)
d7 = p7.dual
print(d7.fmt('\n'))
print('-' * 10)
s7 = p7.sASP
print(s7.fmt('\n'))
print('#' * 3, a, '#' * 3)
solve(p7,  a)
print('#' * 3, b, '#' * 3)
solve(p7,  b)
print('#' * 3, c, '#' * 3)
solve(p7,  c)
print('#' * 3, a, b, '#' * 3)
solve(p7,  a, b)
print('#' * 3, a, c, '#' * 3)
solve(p7,  a, c)
print('#' * 3, b, c, '#' * 3)
solve(p7,  b, c)
print('#' * 3, a, b, c, '#' * 3)
solve(p7, a, b, c)

p8 = Program(rules=(
    NormalRule(a, (b,)),
    NormalRule(b, (a,))
))
s8 = p8.sASP
print(s8.fmt('\n'))
print('#' * 3, '#' * 3)
solve(p8)
print('#' * 3, a, '#' * 3)
solve(p8, a)
print('#' * 3, b, '#' * 3)
solve(p8, b)
print('#' * 3, a,b, '#' * 3)
solve(p8, a, b)

p9 = Program(rules=(
    NormalRule(a, (b,)),
    NormalRule(b, (a,)),
    NormalRule(c, ()),
))
s9 = p9.sASP
print(s9.fmt('\n'))
print('#' * 3, '#' * 3)
solve(p9)
print('#' * 3, a, '#' * 3)
solve(p9, a)
print('#' * 3, b, '#' * 3)
solve(p9, b)
print('#' * 3, a, b, '#' * 3)
solve(p9, a, b)
print('#' * 3, c, a, b, '#' * 3)
solve(p9, c, a, b)
