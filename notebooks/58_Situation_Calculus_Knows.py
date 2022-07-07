from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cache
from typing import FrozenSet, Set, Optional, Iterator, Mapping, TypeVar, Union, Callable, Sequence, Tuple

import more_itertools

#%%

def powerset(iterable):
    for i in more_itertools.powerset(iterable):
        yield set(i)


def frozen_powerset(iterable):
    for i in more_itertools.powerset(iterable):
        yield frozenset(i)
#%%

_ForwardClassicalAtom = TypeVar('_ForwardClassicalAtom', bound='ClassicalAtom')


@dataclass(frozen=True, order=True)
class ClassicalAtom:
    symbol: str
    arguments: Sequence[_ForwardClassicalAtom] = field(default_factory=tuple)

    def __neg__(self):
        if self.is_complement:
            return ClassicalAtom(self.symbol[1:])
        else:
            return ClassicalAtom('-{}'.format(self.symbol))

    def __abs__(self):
        if self.is_complement:
            return -self
        return self

    def __str__(self):
        if self.arguments:
            return "{}({})".format(self.symbol, ','.join(map(str, self.arguments)))
        else:
            return self.symbol

    @property
    def is_complement(self) -> bool:
        return self.symbol.startswith('-')


ClassicalAlphabet = Set[ClassicalAtom]
ClassicalValuation = Mapping[ClassicalAtom, bool]


@dataclass(frozen=True, order=True)
class ClassicalLiteral:
    atom: ClassicalAtom
    sign: bool = field(default=True)

    def __str__(self):
        sign_str = ""
        if not self.sign:
            sign_str = "¬"
        return "{}{}".format(sign_str, self.atom)

    def __repr__(self):
        return str(self)

    def __neg__(self):
        return ClassicalLiteral(self.atom, not self.sign)

    def __invert__(self):
        return ClassicalLiteral(-self.atom, self.sign)

    def __and__(self, other):
        left = ClassicalFormula(self)
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        return ClassicalFormula(left, ClassicalConnective.And, right)

    def __or__(self, other):
        left = ClassicalFormula(self)
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        return ClassicalFormula(left, ClassicalConnective.Or, right)

    def __rshift__(self, other):
        left = ClassicalFormula(self)
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        return ClassicalFormula(left, ClassicalConnective.Implies, right)

    def __call__(self, valuation: Optional[ClassicalValuation] = None, *args, **kwargs):
        return self.evaluate(valuation)

    def evaluate(self, valuation: Optional[ClassicalValuation] = None):
        if valuation is None:
            return isinstance(self, ClassicalTop)
        return self.sign == valuation.get(self.atom, False)


@dataclass(frozen=True, order=True)
class ClassicalTop(ClassicalLiteral):
    atom: ClassicalAtom = field(default=ClassicalAtom('⊤'), init=False)
    sign: bool = field(default=True, init=False)

    def __str__(self):
        return str(self.atom)

    def __repr__(self):
        return str(self)

    def __neg__(self):
        return ClassicalBot()

    def __and__(self, other):
        if isinstance(other, ClassicalLiteral):
            return ClassicalFormula(other)
        return other

    def __or__(self, other):
        return ClassicalFormula(ClassicalTop())


@dataclass(frozen=True, order=True)
class ClassicalBot(ClassicalLiteral):
    atom: ClassicalAtom = field(default=ClassicalAtom('⊥'), init=False)
    sign: bool = field(default=False, init=False)

    def __neg__(self):
        return ClassicalTop()

    def __str__(self):
        return str(self.atom)

    def __and__(self, other):
        return ClassicalFormula(ClassicalBot())

    def __or__(self, other):
        if isinstance(other, ClassicalLiteral):
            return ClassicalFormula(other)
        return other


class ClassicalConnective(IntEnum):
    And = 0
    Or = 1
    Implies = 2

    def __str__(self):
        if self is ClassicalConnective.And:
            return "∧"
        elif self is ClassicalConnective.Or:
            return "∨"
        elif self is ClassicalConnective.Implies:
            return "→"
        else:
            assert False, "Unhandled Connective.__str__: {} = {}".format(self.name, self.value)

    def evaluate(self, left: bool, right: bool):
        if self is ClassicalConnective.And:
            return left and right
        elif self is ClassicalConnective.Or:
            return left or right
        elif self is ClassicalConnective.Implies:
            return not left or right
        else:
            assert False, "Unhandled Connective.evaluate: {} = {}".format(self.name, self.value)


_ForwardClassicalFormula = TypeVar('_ForwardClassicalFormula', bound='ClassicalFormula')


@dataclass(frozen=True, order=True)
class ClassicalFormula:
    left: Union[_ForwardClassicalFormula, ClassicalLiteral]
    connective: Optional[ClassicalConnective] = field(default=None)
    right: Union[_ForwardClassicalFormula, None] = field(default=None)

    def __str__(self):
        left_str = str(self.left)
        connective_str = ""
        if self.connective is not None:
            connective_str = " {}".format(self.connective)
            if isinstance(self.left,
                          ClassicalFormula) and self.left.connective is not None and self.left.connective > self.connective:
                left_str = "({})".format(left_str)
        right_str = ""
        if self.right is not None:
            if self.right.left == ClassicalBot() and self.right.right is None and self.connective is ClassicalConnective.Implies:
                left_str = "¬({})".format(left_str)
                connective_str = ""
            else:
                right_str = " {}".format(self.right)
            if isinstance(self.right,
                          ClassicalFormula) and self.right.connective is not None and self.right.connective > self.connective:
                left_str = "({})".format(left_str)
        return "{}{}{}".format(left_str, connective_str, right_str)

    def __repr__(self):
        return str(self)

    def __neg__(self):
        if self.connective is not None and self.right is None:
            raise TypeError("Formula.connective present, despite Formula.right missing.")
        elif self.connective is None and self.right is not None:
            raise TypeError("Formula.connective missing, despite Formula.right present.")

        if self.connective is None and self.right is None:
            return ClassicalFormula(-self.left)
        elif self.connective is ClassicalConnective.And:
            return ClassicalFormula(-self.left, ClassicalConnective.Or, -self.right)
        elif self.connective is ClassicalConnective.Or:
            return ClassicalFormula(-self.left, ClassicalConnective.And, -self.right)
        elif self.connective is ClassicalConnective.Implies:
            if self.right.left == ClassicalBot() and self.right.right is None:
                return self.left
            return ClassicalFormula(self, ClassicalConnective.Implies, ClassicalFormula(ClassicalBot()))
        else:
            assert False, "Unknown Formula.connective. {} = {}.".format(self.connective.name, self.connective.value)

    def __and__(self, other):
        left = self
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        if left.is_top:
            return right
        elif right.is_top:
            return left
        if left.is_bot:
            return left
        elif right.is_bot:
            return right
        return ClassicalFormula(left, ClassicalConnective.And, right)

    def __or__(self, other):
        left = self
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        if left.is_top:
            return left
        elif right.is_top:
            return right
        if left.is_bot:
            return right
        elif right.is_bot:
            return left
        return ClassicalFormula(left, ClassicalConnective.Or, right)

    def __rshift__(self, other):
        left = self
        right = other
        if isinstance(other, ClassicalLiteral):
            right = ClassicalFormula(right)
        return ClassicalFormula(left, ClassicalConnective.Implies, right)

    def __call__(self, valuation: Optional[ClassicalValuation] = None) -> bool:
        return self.evaluate(valuation)

    @property
    def literals(self) -> Set[ClassicalLiteral]:
        literals = set()
        if isinstance(self.left, ClassicalLiteral):
            if not isinstance(self.left, ClassicalTop) and not isinstance(self.left, ClassicalBot):
                literals.add(self.left)
        else:
            assert isinstance(self.left, ClassicalFormula), "Unknown type for Formula.right. {}: {}".format(
                type(self.left).__name__, self.left)
            literals.update(self.left.literals)
        if self.right is not None:
            assert isinstance(self.right, ClassicalFormula), "Unknown type for Formula.right. {}: {}".format(
                type(self.right).__name__, self.right)
            literals.update(self.right.literals)
        return literals

    @property
    def atoms(self) -> Set[ClassicalAtom]:
        return {literal.atom for literal in self.literals}

    @property
    def is_top(self) -> bool:
        return self.is_literal and isinstance(self.left, ClassicalTop)

    @property
    def is_bot(self) -> bool:
        return self.is_literal and isinstance(self.left, ClassicalBot)

    @property
    def is_literal(self) -> bool:
        return self.right is None

    def evaluate(self, valuation: Optional[ClassicalValuation] = None) -> bool:
        if isinstance(self.left, ClassicalLiteral):
            value_left = self.__evaluate_literal(self.left, valuation)
        else:
            assert isinstance(self.left, ClassicalFormula), "Unknown type for Formula.left. {}: {}".format(
                type(self.left).__name__, self.left)
            value_left = self.left.evaluate(valuation)
        if self.connective is not None and self.right is None:
            raise TypeError("Formula.connective present, despite Formula.right missing.")
        elif self.connective is None and self.right is not None:
            raise TypeError("Formula.connective missing, despite Formula.right present.")

        if self.connective is None and self.right is None:
            return value_left
        else:
            assert isinstance(self.right, ClassicalFormula), "Unknown type for Formula.right. {}: {}".format(
                type(self.right).__name__, self.right)
            value_right = self.right.evaluate(valuation)

            return self.connective.evaluate(value_left, value_right)

    def __evaluate_literal(self, literal: ClassicalLiteral, valuation: Optional[ClassicalValuation] = None) -> bool:
        if isinstance(literal, ClassicalTop) or isinstance(literal, ClassicalBot):
            return literal.sign
        else:
            # get assigned truth value of atom (per default false) and flip the result if negated
            return valuation is not None and bool(valuation.get(literal.atom, False) ^ (not literal.sign))

    def set_to_bot(self, *atoms: ClassicalAtom) -> _ForwardClassicalFormula:
        if isinstance(self.left, ClassicalLiteral):
            assert self.connective is None
            assert self.right is None
            if self.left.atom in atoms:
                if self.left.sign:
                    return ClassicalFormula(ClassicalBot())
                else:
                    return ClassicalFormula(ClassicalTop())
            else:
                return self
        else:
            left = self.left.set_to_bot(*atoms)
            right = self.right
            if right is not None:
                right = self.right.set_to_bot(*atoms)
            return ClassicalFormula(left, self.connective, right)

#%%
def all_valuations(alphabet: ClassicalAlphabet, complete: bool = False) -> Iterator[ClassicalValuation]:
    subsets = powerset(alphabet)
    for subset in subsets:
        valuation = defaultdict(lambda: False)
        for atom in subset:
            valuation[atom] = True
        if complete:
            for atom in alphabet:
                if atom not in subset:
                    valuation[atom] = False
        yield valuation


def models(formulas: Set[ClassicalFormula], alphabet: Optional[ClassicalAlphabet] = None) -> Iterator[
    ClassicalValuation]:
    if alphabet is None:
        alphabet = {atom for formula in formulas for atom in formula.atoms}
    for valuation in all_valuations(alphabet):
        if all(formula.evaluate(valuation) for formula in formulas):
            yield valuation


def sat(formulas: Set[ClassicalFormula], alphabet: Optional[ClassicalAlphabet] = None) -> bool:
    model = next(models(formulas, alphabet), None)
    return model is not None


def unsat(formulas: Set[ClassicalFormula], alphabet: Optional[ClassicalAlphabet] = None) -> bool:
    return not sat(formulas, alphabet)


def entails(formulas: Set[ClassicalFormula], formula: ClassicalFormula) -> bool:
    return unsat(formulas | {-formula})


def valid(formulas: Set[ClassicalFormula], alphabet: Optional[ClassicalAlphabet] = None) -> bool:
    if alphabet is None:
        alphabet = {atom for formula in formulas for atom in formula.atoms}
    for valuation in all_valuations(alphabet):
        if any(not formula.evaluate(valuation) for formula in formulas):
            return False
    return True
#%%
Action = ClassicalAtom
Fluent = ClassicalAtom
FluentLiteral = ClassicalLiteral
#%%
ActionAlphabet = ClassicalAlphabet
FluentAlphabet = ClassicalAlphabet
Time = int
#%%
_ForwardState = TypeVar('_ForwardState', bound='State')


@dataclass(frozen=True)
class State:
    _state: FrozenSet[FluentLiteral]
    parent: Optional[_ForwardState] = field(default=None)
    action: Optional[Action] = field(default=None)

    @property
    def coherent(self) -> bool:
        return all(-literal not in self for literal in self)

    @property
    def time(self) -> int:
        count = 0
        node = self
        while node.parent is not None:
            count += 1
            node = node.parent
        return count

    def __iter__(self):
        return iter(self._state)

    def __str__(self):
        return "{}{}{}${}".format('{', ','.join(map(str, sorted(self))), '}', self.action)

    def __sub__(self, other):
        return State(self._state - other._state, self.parent, self.action)

    def __or__(self, other):
        return State(self._state | other._state, self.parent, self.action)

    def __len__(self):
        return len(self._state)

    def __call__(self, causal_setting, action: Action, *args, **kwargs):
        return self.apply(causal_setting, action)

    def reachable(self, causal_setting, other) -> bool:
        assert isinstance(other, State)
        return self != other and other in all_states(causal_setting=causal_setting, state=self, upto_time=other.time)

    def reachable_or_eq(self, causal_setting, other) -> bool:
        assert isinstance(other, State)
        return self == other or self.reachable(causal_setting, other)

    def legally_reachable(self, causal_setting, other) -> bool:
        assert isinstance(other, State)
        return self.reachable(causal_setting, other) and other.executable(causal_setting)

    def executable(self, causal_setting) -> bool:
        return True

    def legally_reachable_or_eq(self, causal_setting, other) -> bool:
        assert isinstance(other, State)

        return self.reachable_or_eq(causal_setting, other) and other.executable(causal_setting)

    def complete(self, causal_setting) -> bool:
        fluents, _, _, _, _, _ = causal_setting
        return len(fluents) == len(self)

    def as_valuation(self) -> ClassicalValuation:
        return defaultdict(lambda: False, {literal.atom: literal.sign for literal in self})

    def apply(self, causal_setting, action: Action) -> _ForwardState:
        _, _, poss, do, _, _ = causal_setting
        return do(action, self)


#%%
SituationStep = Tuple[State, Action, State]
Trace = Sequence[SituationStep]
Poss = Callable[[Action, State], bool]
Do = Callable[[Action, State], State]
InitialState = State
K_Relation = Mapping[State, FrozenSet[State]]
#%%
CausalSetting = Tuple[FluentAlphabet, ActionAlphabet, Poss, Do, InitialState, Time]
#%%
_T = TypeVar('_T')
_U = TypeVar('_U')


def mapping_to_callable_rel(m: Mapping[_T, _U]) -> Callable[[_T, _U], bool]:
    def rel(_t: _T, _u: _U) -> bool:
        return m.get(_t, None) == _u

    return rel
#%%
_ForwardDynamicFormula = TypeVar('_ForwardDynamicFormula', bound='DynamicFormula')


class DynamicFormula:

    @property
    def is_predicate(self) -> bool:
        return False

    @property
    def is_poss(self) -> bool:
        return False

    @property
    def is_after(self) -> bool:
        return False

    @property
    def is_negation(self) -> bool:
        return False

    @property
    def is_conjunction(self) -> bool:
        return False

    @property
    def is_existential_quantification(self) -> bool:
        return False

    @property
    def is_know(self) -> bool:
        return False

    def __neg__(self):
        if self.is_negation:
            assert isinstance(self, DynamicNegation)
            return self.dynamic_formula
        return DynamicNegation(self)

    def __and__(self, other):
        return DynamicConjunction(self, other)

    # TODO: Type synonym
    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        raise NotImplementedError


@dataclass(order=True, frozen=True)
class DynamicPredicate(DynamicFormula):
    predicate: ClassicalLiteral

    @property
    def is_predicate(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Optional[Mapping[State, FrozenSet[State]]] = None, causal_setting: Optional[CausalSetting] = None) -> bool:
        return self.predicate in state


@dataclass(order=True, frozen=True)
class DynamicPoss(DynamicFormula):
    action: Action

    @property
    def is_poss(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        _, _, poss, _, _, _ = causal_setting
        return poss(self.action, state)


@dataclass(order=True, frozen=True)
class DynamicAfter(DynamicFormula):
    action: Action
    dynamic_formula: DynamicFormula

    @property
    def is_after(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        _, _, poss, do, _, _ = causal_setting
        return self.dynamic_formula.evaluate(do(self.action, state), k_rel, causal_setting)


@dataclass(order=True, frozen=True)
class DynamicNegation(DynamicFormula):
    dynamic_formula: DynamicFormula

    @property
    def is_negation(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        return not self.dynamic_formula.evaluate(state, k_rel, causal_setting)


@dataclass(order=True, frozen=True)
class DynamicConjunction(DynamicFormula):
    left: DynamicFormula
    right: DynamicFormula

    @property
    def is_conjunction(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        return self.left.evaluate(state, k_rel, causal_setting) and self.right.evaluate(state, k_rel, causal_setting)


@dataclass(order=True, frozen=True)
class DynamicExistentialQuantification(DynamicFormula):
    parameters: FrozenSet[ClassicalAtom]
    dynamic_formula: DynamicFormula

    @property
    def is_existential_quantification(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: Mapping[State, FrozenSet[State]], causal_setting: CausalSetting) -> bool:
        raise NotImplementedError


#%%
def all_states(causal_setting: CausalSetting,
               state: Optional[State] = None,
               from_time: Optional[int] = None,
               upto_time: Optional[int] = None) -> Iterator[State]:
    _, actions, poss, do, initial_state, time = causal_setting
    if state is None:
        state = initial_state
    queue = [state]

    while queue:
        current = queue.pop(0)
        if current.time <= time and (upto_time is None or current.time <= upto_time):
            for action in actions:
                if poss(action, current):
                    state_ = current.apply(causal_setting, action)
                    queue.append(state_)
            if from_time is None or from_time <= current.time:
                yield current


def _causes_directly_gen(causal_setting: CausalSetting, k_rel: K_Relation, action: Action, time: int, dynamic_formula: DynamicFormula,
                         state: State) -> Iterator[State]:
    _, _, poss, do, initial_state, time_range = causal_setting
    for state_ in all_states(causal_setting, from_time=time):
        if not state_.time == time:
            continue
        state__ = state_(causal_setting, action)
        if not initial_state.reachable(causal_setting, state__):
            continue
        if not state__.reachable_or_eq(causal_setting, state):
            continue
        if not (-dynamic_formula).evaluate(state_, k_rel, causal_setting):
            continue
        if not all(dynamic_formula.evaluate(_state_, k_rel, causal_setting) for _state_ in
                   all_states(causal_setting, upto_time=state.time) if
                   state__.reachable_or_eq(causal_setting, _state_) and _state_.reachable_or_eq(causal_setting, state)):
            continue
        yield state_

def causes_directly(causal_setting: CausalSetting, k_rel: K_Relation, action: Action, time: int, dynamic_formula: DynamicFormula,
                    state: State) -> bool:
    return next(_causes_directly_gen(causal_setting, k_rel, action, time, dynamic_formula, state), None) is not None

#example_4_knows_1.evaluate(state=example_3_sigma, k_rel=example_4_k_rel, causal_setting=example_4_causal_setting)
#%%
b1 = ClassicalAtom('b1')
b2 = ClassicalAtom('b2')

pickUp_b1 = ClassicalAtom('pickUp', (b1,))
pickUp_b2 = ClassicalAtom('pickUp', (b2,))

drop_b1 = ClassicalAtom('drop', (b1,))
drop_b2 = ClassicalAtom('drop', (b2,))

holding_b1 = ClassicalAtom('holding', (b1,))
holding_b2 = ClassicalAtom('holding', (b2,))

l_holding_b1 = ClassicalLiteral(holding_b1)
l_holding_b2 = ClassicalLiteral(holding_b2)

#%%
example_1_fluents = frozenset({holding_b1, holding_b2})
example_1_actions = frozenset({pickUp_b1, pickUp_b2, drop_b1, drop_b2})

def example_1_do(action: Action, state: State) -> State:
    box = action.arguments[0]
    l_holding = ClassicalLiteral(ClassicalAtom('holding', (box,)))
    if action.symbol == 'pickUp':
        _state_ = (state._state - {-l_holding}) | {l_holding}
        state_ = State(_state_, parent=state, action=action)
        return state_
    elif action.symbol == 'drop':
        _state_ = (state._state - {l_holding}) | {-l_holding}
        state_ = State(_state_, parent=state, action=action)
        return state_
    else:
        assert False

def example_1_poss(action: Action, state: State) -> bool:
    box = action.arguments[0]
    l_holding = ClassicalLiteral(ClassicalAtom('holding', (box,)))
    if action.symbol == 'pickUp':
        return not any(literal.atom.symbol == 'holding' for literal in state if literal.sign)
    elif action.symbol == 'drop':
        return l_holding in state
    else:
        return False

example_1_initial_state = State(frozenset({l_holding_b2, -l_holding_b1}))
example_1_causal_setting = (example_1_fluents, example_1_actions, example_1_poss, example_1_do, example_1_initial_state, 4)
#%%
example_1_initial_state.time
#%%
example_1_state = example_1_initial_state.apply(example_1_causal_setting, drop_b2)
print(example_1_state.time)
example_1_state
#%%
example_1_states = all_states(example_1_causal_setting)
for state in example_1_states:
    print(state)
#%%
example_1_state = example_1_initial_state(example_1_causal_setting, drop_b2)(example_1_causal_setting, pickUp_b2)
print(example_1_state, example_1_state.time)
#%%
causes_directly(causal_setting=example_1_causal_setting, k_rel={}, action=pickUp_b2, time=1, dynamic_formula=DynamicPredicate(l_holding_b2), state=example_1_state)
#%%
broken_b1 = ClassicalAtom('broken', (b1,))
broken_b2 = ClassicalAtom('broken', (b2,))

l_broken_b1 = ClassicalLiteral(broken_b1)
l_broken_b2 = ClassicalLiteral(broken_b2)

fragile_b1 = ClassicalAtom('fragile', (b1,))
fragile_b2 = ClassicalAtom('fragile', (b2,))

l_fragile_b1 = ClassicalLiteral(fragile_b1)
l_fragile_b2 = ClassicalLiteral(fragile_b2)

quench_b1 = ClassicalAtom('quench', (b1,))
quench_b2 = ClassicalAtom('quench', (b2,))
#%%
example_2_fluents = frozenset({holding_b1, holding_b2, fragile_b1, fragile_b2})
example_2_actions = frozenset({pickUp_b1, pickUp_b2, drop_b1, drop_b2, quench_b1, quench_b2})

def example_2_do(action: Action, state: State) -> State:
    box = action.arguments[0]
    l_holding = ClassicalLiteral(ClassicalAtom('holding', (box,)))
    l_fragile = ClassicalLiteral(ClassicalAtom('fragile', (box,)))
    l_broken = ClassicalLiteral(ClassicalAtom('broken', (box,)))
    if action.symbol == 'pickUp':
        _state_ = (state._state - {-l_holding}) | {l_holding}
    elif action.symbol == 'drop':
        _state_ = (state._state - {l_holding}) | {-l_holding}
        if l_fragile in _state_:
            _state_ = (_state_ - {-l_broken}) | {l_broken}
    elif action.symbol == 'quench':
        _state_ = (state._state - {-l_fragile}) | {l_fragile}
    else:
        assert False
    state_ = State(_state_, parent=state, action=action)
    return state_

def example_2_poss(action: Action, state: State) -> bool:
    box = action.arguments[0]
    l_holding = ClassicalLiteral(ClassicalAtom('holding', (box,)))
    if action.symbol == 'pickUp':
        return not any(literal.atom.symbol == 'holding' for literal in state if literal.sign)
    elif action.symbol == 'drop':
        return l_holding in state
    elif action.symbol == 'quench':
        return True
    else:
        return False

example_2_initial_state = State(frozenset({l_holding_b2, -l_holding_b1, -l_fragile_b2, -l_fragile_b1, -l_broken_b2, -l_broken_b1}))
example_2_causal_setting = (example_2_fluents, example_2_actions, example_2_poss, example_2_do, example_2_initial_state, 4)
example_2_causal_setting
#%%
print(example_2_initial_state)
#%%
example_2_state_ = example_2_initial_state(example_2_causal_setting, quench_b2)
print(example_2_state_)
causes_directly(example_2_causal_setting, {}, quench_b2, 0, DynamicPredicate(l_fragile_b2), example_2_state_)
#%%
example_2_state__ = example_2_state_(example_2_causal_setting, drop_b2)
print(example_2_state__)
causes_directly(example_2_causal_setting, {}, drop_b2, 1, DynamicPredicate(l_broken_b2), example_2_state__)
#%%
def causes_indirectly(causal_setting: CausalSetting, k_rel: K_Relation, action: Action, time: Time, dynamic_formula: DynamicFormula, state: State) -> bool:
    print()
    print(action, time, dynamic_formula)
    print()
    _, actions, poss, _, _, _ = causal_setting
    for state_ in all_states(causal_setting, upto_time=state.time):
        if state_.reachable(causal_setting, state):
            for action_ in actions:
                if not causes_directly(causal_setting, k_rel, action_, state_.time, dynamic_formula, state):
                    continue
                if causes_directly(causal_setting, k_rel, action, time, DynamicConjunction(DynamicPoss(action_), DynamicAfter(action_, dynamic_formula)), state_):
                    return True
                if causes_indirectly(causal_setting, k_rel, action, time, DynamicConjunction(DynamicPoss(action_), DynamicAfter(action_, dynamic_formula)), state_):
                    return True
    return False

#%%
def causes(causal_setting: CausalSetting, k_rel: K_Relation, action: Action, time: Time, dynamic_formula: DynamicFormula, state: State) -> bool:
    return causes_directly(causal_setting, k_rel, action, time, dynamic_formula, state) or causes_indirectly(causal_setting, k_rel, action, time, dynamic_formula, state)

#causes(example_3_causal_setting, drop_b1, 0, example_3_phi, example_3_sigma)
#%%
print(example_2_state_)
causes(example_2_causal_setting, {}, quench_b2, 0, DynamicPredicate(l_fragile_b2), example_2_state_)
#%%
print(example_2_state__)
causes(example_2_causal_setting, {}, drop_b2, 1, DynamicPredicate(l_broken_b2), example_2_state__)
#%%
print(example_2_state__)
causes_directly(example_2_causal_setting, {}, quench_b2, 0, DynamicPredicate(l_broken_b2), example_2_state__)
#%%
print(example_2_state__)
causes(example_2_causal_setting, {}, quench_b2, 0, DynamicPredicate(l_broken_b2), example_2_state__)
#%%
example_3_fluents = frozenset({holding_b1, holding_b2, fragile_b1, fragile_b2})
example_3_actions = frozenset({pickUp_b1, pickUp_b2, drop_b1, drop_b2, quench_b1, quench_b2})

example_3_do = example_2_do

example_3_poss = example_2_poss

example_3_initial_state = State(frozenset({-l_holding_b2, l_holding_b1, -l_fragile_b2, -l_fragile_b1, -l_broken_b2, -l_broken_b1}))
example_3_causal_setting = (example_3_fluents, example_3_actions, example_3_poss, example_3_do, example_3_initial_state, 6)
example_3_causal_setting
#%%
# example_3
example_3_sigma = example_3_initial_state(example_3_causal_setting, drop_b1)(example_3_causal_setting, quench_b1)(example_3_causal_setting, quench_b2)(example_3_causal_setting, pickUp_b1)(example_3_causal_setting, drop_b1)
print(example_3_sigma)
#%%
example_3_phi = DynamicPredicate(l_broken_b1)
#%%
print(example_3_sigma)
print(example_3_phi)
#%%
causes_directly(example_3_causal_setting, {}, drop_b1, 0, example_3_phi, example_3_sigma)
#%%
#causes(example_3_causal_setting, {}, drop_b1, 0, example_3_phi, example_3_sigma)
#%%
causes_directly(example_3_causal_setting, {}, quench_b1, 1, example_3_phi, example_3_sigma)
#%%
#causes(example_3_causal_setting, {}, quench_b1, 1, example_3_phi, example_3_sigma)
#%%
causes_directly(example_3_causal_setting, {}, pickUp_b1, 3, example_3_phi, example_3_sigma)
#%%
#causes(example_3_causal_setting, {}, pickUp_b1, 3, example_3_phi, example_3_sigma)
#%%
causes_directly(example_3_causal_setting, {}, drop_b1, 4, example_3_phi, example_3_sigma)
#%%
#causes(example_3_causal_setting, {}, drop_b1, 4, example_3_phi, example_3_sigma)
#%%
causes_directly(example_3_causal_setting, {}, quench_b2, 2, example_3_phi, example_3_sigma)
#%%
#causes(example_3_causal_setting, {}, quench_b2, 2, example_3_phi, example_3_sigma)
#%%
@dataclass(order=True, frozen=True)
class DynamicCauses(DynamicFormula):
    action: Action
    time: Time
    dynamic_formula: DynamicFormula

    def evaluate(self, state: State, k_rel: K_Relation, causal_setting: CausalSetting) -> bool:
        return causes(causal_setting, k_rel, self.action, self.time, self.dynamic_formula, state)


#%%
@dataclass(order=True, frozen=True)
class DynamicKnows(DynamicFormula):
    dynamic_formula: DynamicFormula

    @property
    def is_know(self) -> bool:
        return True

    def evaluate(self, state: State, k_rel: K_Relation, causal_setting: CausalSetting) -> bool:
        for state_,known_states in k_rel.items():
            if state in known_states:
                #causal_setting_ = causal_setting[0], causal_setting[1], causal_setting[2], causal_setting[3], state_, causal_setting[5]
                if not self.dynamic_formula.evaluate(state_, k_rel, causal_setting):
                    return False
        return True

#%%
example_4_fluents = frozenset({holding_b1, holding_b2, fragile_b1, fragile_b2})
example_4_actions = frozenset({pickUp_b1, pickUp_b2, drop_b1, drop_b2, quench_b1, quench_b2})

example_4_do = example_2_do

example_4_poss = example_2_poss



example_4_initial_state = State(frozenset({l_holding_b1, -l_holding_b2, -l_broken_b1, -l_broken_b2, -l_fragile_b1, -l_fragile_b1, -l_fragile_b2}))

example_4_initial_state_1 = State(frozenset({l_holding_b1, -l_holding_b2, -l_broken_b1, -l_broken_b2, -l_fragile_b1, l_fragile_b1, l_fragile_b2}))
example_4_initial_state_2 = State(frozenset({l_holding_b1, -l_holding_b2, -l_broken_b1, -l_broken_b2, -l_fragile_b1, l_fragile_b1, -l_fragile_b2}))
example_4_initial_state_3 = State(frozenset({l_holding_b1, -l_holding_b2, -l_broken_b1, -l_broken_b2, -l_fragile_b1, -l_fragile_b1, l_fragile_b2}))

example_4_initial_states = frozenset({example_4_initial_state, example_4_initial_state_1, example_4_initial_state_2, example_4_initial_state_3})

example_4_k_rel = {
    example_4_initial_state: example_4_initial_states,
    example_4_initial_state_1: example_4_initial_states,
    example_4_initial_state_2: example_4_initial_states,
    example_4_initial_state_3: example_4_initial_states,
}

example_4_causal_setting:CausalSetting = (example_4_fluents, example_4_actions, example_4_poss, example_4_do, example_4_initial_state, 6)
example_4_causal_setting
#%%

example_4_actions_sequence = [drop_b1, quench_b1, quench_b2, pickUp_b1, drop_b1]
last = deepcopy(example_4_k_rel)
for action in example_4_actions_sequence:
    next_ = {}
    for state,known_states in last.items():
        next_[state(example_4_causal_setting, action)] = frozenset(known_state(example_4_causal_setting, action) for known_state in known_states if example_4_poss(action, known_state))
    example_4_k_rel |= next_
    last = next_

for state,known_states in example_4_k_rel.items():
    print(state)
    print('-'*8)
    for known_state in known_states:
        print(known_state)
    print('#'*8)
print(type(example_4_k_rel).__name__)



example_4_knows_1 = DynamicKnows(DynamicCauses(drop_b1, 0, example_3_phi))
example_4_knows_1.evaluate(state=example_3_sigma, k_rel=example_4_k_rel, causal_setting=example_4_causal_setting)
#%%

#%%
example_4_knows_2 = DynamicKnows(DynamicNegation(DynamicCauses(quench_b2, 2, example_3_phi)))
example_4_knows_2.evaluate(example_3_sigma, example_4_k_rel, example_4_causal_setting)
#%%
@dataclass(order=True, frozen=True)
class DynamicKWhether(DynamicFormula):
    dynamic_formula: DynamicFormula

    def evaluate(self, state: State, k_rel: K_Relation, causal_setting: CausalSetting) -> bool:
        return DynamicKnows(self.dynamic_formula).evaluate(state, k_rel, causal_setting) or DynamicKnows(-self.dynamic_formula).evaluate(state, k_rel, causal_setting)
#%%

example_4_whether_1 = DynamicKWhether(DynamicCauses(quench_b1, 1, example_3_phi))
example_4_whether_1.evaluate(example_3_sigma, example_4_k_rel, example_4_causal_setting)
#%%
example_4_whether_2 = DynamicKWhether(DynamicCauses(pickUp_b1, 3, example_3_phi))
example_4_whether_2.evaluate(example_3_sigma, example_4_k_rel, example_4_causal_setting)
#%%
example_4_whether_3 = DynamicKWhether(DynamicCauses(drop_b1, 4, example_3_phi))
example_4_whether_3.evaluate(example_3_sigma, example_4_k_rel, example_4_causal_setting)