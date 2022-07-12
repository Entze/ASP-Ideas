from dataclasses import dataclass, field
from functools import cache
from typing import TypeVar, Sequence, FrozenSet, Optional, Collection, Set, Union, Tuple, Iterator

import more_itertools

#%%
_U = TypeVar('_U')
_ForwardFrozenRelation = TypeVar('_ForwardFrozenRelation', bound='FrozenRelation')


@dataclass(frozen=True)
class FrozenRelation:
    relation_set: FrozenSet[tuple] = field(default_factory=frozenset)

    def __getitem__(self, item):
        return self.get(item)

    def __str__(self):
        return "{}{}{}".format('{', ",".join(map(lambda e: "({})".format(','.join(map(str, e))), self.relation_set)), '}')

    def related(self, *items) -> bool:
        return items in self.relation_set

    def get(self, item) -> Set[_U]:
        return {elem[1] for elem in self.relation_set if elem[0] == item}

    def get_inverse(self, item) -> Set[_U]:
        return {elem[0] for elem in self.relation_set if elem[1] == item}

    @staticmethod
    def from_ground_coll(ground_coll: Collection[tuple],
                         transitive: bool = False,
                         reflexive: bool = False,
                         symmetric: bool = False) -> _ForwardFrozenRelation:
        relation_set = set(ground_coll)
        add = {None}
        while add:
            add = set()
            if symmetric:
                for rel in relation_set:
                    item1, item2 = rel
                    if (item2, item1) not in relation_set:
                        add.add((item2, item1))
            if reflexive:
                for rel in relation_set:
                    item1, item2 = rel
                    if (item1, item1) not in relation_set:
                        add.add((item1, item1))
                    if (item2, item2) not in relation_set:
                        add.add((item2, item2))
            if transitive:
                for rel in relation_set:
                    item1, item2 = rel
                    for rel_ in relation_set:
                        item3, item4 = rel_
                        if item2 != item3:
                            continue
                        if (item1, item4) not in relation_set:
                            add.add((item1, item4))
            relation_set.update(add)
        return FrozenRelation(frozenset(relation_set))

#%%
_ForwardAtom = TypeVar('_ForwardAtom', bound='Atom')
_ForwardLiteral = TypeVar('_ForwardLiteral', bound='Literal')
_ForwardFormula = TypeVar('_ForwardFormula', bound='Formula')


@dataclass(order=True, frozen=True)
class Atom:
    symbol: str
    arguments: Sequence[Union[_ForwardAtom, _ForwardFormula, _ForwardLiteral]] = field(default_factory=tuple)

    @property
    def is_complement(self) -> bool:
        return self.symbol.startswith('-')

    @property
    def is_know(self) -> bool:
        return self.symbol == 'Know'

    def __invert__(self):
        if self.is_complement:
            return Atom(self.symbol[1:], self.arguments)
        else:
            return Atom('-{}'.format(self.symbol), self.arguments)

    def __str__(self):
        if self.arguments:
            return "{}({})".format(self.symbol, ','.join(map(str, self.arguments)))
        else:
            return "{}".format(self.symbol)

#%%
@dataclass(order=True, frozen=True)
class Literal:
    atom: Atom
    sign: bool = field(default=True)

    def __neg__(self):
        return Literal(self.atom, not self.sign)

    def __abs__(self):
        return Literal(self.atom)

    def __invert__(self):
        return Literal(~self.atom, self.sign)

    def __str__(self):
        if self.sign:
            return str(self.atom)
        else:
            return "¬{}".format(self.atom)
#%%
Action = Atom
Fluent = Atom
FluentLiteral = Literal
#%%
_ForwardCausalSetting = TypeVar('_ForwardCausalSetting', bound='CausalSetting')
#%%
@dataclass(order=True, frozen=True)
class State:
    world: FrozenSet[FluentLiteral] = field(default_factory=frozenset)
    k_relation: FrozenRelation = field(default_factory=FrozenRelation)
    prioritized_goals: Sequence[_ForwardFormula] = field(default_factory=tuple)

    def __contains__(self, item):
        if not isinstance(item, FluentLiteral):
            raise TypeError("Unexpected type {} of item {}".format(type(item).__name__, item))
        if item in self.world:
            return True
        if -item in self.world:
            return False
        return not item.sign

#%%
_ForwardSituation = TypeVar('_ForwardSituation', bound='Situation')


@dataclass(order=True, frozen=True)
class Situation:
    state: State = field(default_factory=State)
    action: Optional[Action] = field(default=None)
    parent: Optional[_ForwardSituation] = field(default=None, repr=False)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def time(self) -> int:
        if self.is_root:
            return 0
        else:
            return self.parent.time + 1

    @property
    def root(self) -> _ForwardSituation:
        if self.is_root:
            return self
        else:
            return self.parent.root
#%%
_ForwardPath = TypeVar('_ForwardPath', bound='Path')


@dataclass(order=True, frozen=True)
class Path:
    states: Sequence[State] = field(default_factory=tuple) # TODO: Change State -> Sequence
    actions: Sequence[Action] = field(default_factory=tuple)

    @property
    def root(self) -> State:
        return self.states[0]

    def time(self, state: State) -> Optional[int]:
        for time, state_ in enumerate(self.states):
            if state_ == state:
                return time
        return None

    def starts(self, state: State) -> bool:
        return self.root == state

    def get_suffix(self, state: Optional[State] = None) -> _ForwardPath:
        if state is None:
            state = self.root
        if state not in self.states:
            return Path()
        offset = self.time(state)
        if offset > 0:
            return Path(self.states[offset:], self.actions[offset:])
        else:
            return Path(self.states[offset:], )

#%%
class Formula:
    def __neg__(self):
        # -Formula
        return Negation(self)

    def __and__(self, other):
        # Formula & Formula
        return Conjunction(self, other)

    def __or__(self, other):
        # Formula | Formula
        return Disjunction(self, other)

    def evaluate(self, causal_setting: _ForwardCausalSetting, elem: Union[Situation, Path]) -> bool:
        if isinstance(elem, Situation):
            return self.evaluate_situation(causal_setting, elem)
        else:
            if not isinstance(elem, Path):
                raise TypeError('Unexpected type {} for elem {}'.format(type(elem).__name__, elem))
            return self.evaluate_path(causal_setting, elem)

    def evaluate_situation(self, causal_setting: _ForwardCausalSetting, situation: Situation) -> bool:
        raise NotImplementedError

    def evaluate_path(self, causal_setting: _ForwardCausalSetting, path: Path) -> bool:
        raise NotImplementedError


#%%
class StateFormula(Formula):

    def evaluate_path(self, causal_setting: _ForwardCausalSetting, path: Path) -> bool:
        return self.evaluate_situation(causal_setting, Situation(path.root))

#%%
class PathFormula(Formula):

    # TODO: evaluate_state ?

    pass
#%%
@dataclass(order=True, frozen=True)
class CausalSetting:
    fluent_alphabet: FrozenSet[Fluent] = field(default_factory=frozenset)  # without Poss, Know, Int
    action_alphabet: FrozenSet[Action] = field(default_factory=frozenset)
    initial_state: State = field(default_factory=State)
    k_relation: FrozenRelation = field(default_factory=FrozenRelation)
    max_time: int = field(default=0)

    @cache
    def poss(self, action: Action, situation: Situation) -> bool:
        if action.symbol == 'sense':
            return action.arguments[0] in situation.state
        elif action.symbol == 'req':
            return False
        else:
            raise NotImplementedError

    @cache
    def legal_actions(self, situation: Situation) -> Iterator[Action]:
        return (action for action in self.action_alphabet if self.poss(action, situation))

    @cache
    def _legal_actions(self, state: State) -> Iterator[Action]:
        return (action for action in self.action_alphabet if self.poss(action, Situation(state)))

    def all_paths(self, situation: Situation) -> Iterator[Path]:
        for path in self._all_paths(situation.state):
            states = [situation.state]
            actions = []
            for sa in path:
                state, action = sa
                states.append(state)
                actions.append(action)
            yield Path(tuple(states), tuple(actions))


    def _all_paths(self, state: State, depth: int = 0) -> Iterator[Sequence[Tuple[State, Action]]]:
        if depth <= self.max_time:
            for action in self._legal_actions(state):
                state_ = self._do(action, state)
                all_paths = self._all_paths(state_, depth + 1)
                for path in all_paths:
                    yield ((state_, action), *path)
        else:
            yield ()

    @cache
    def do(self, action: Action, situation: Optional[Situation] = None) -> Situation:
        if situation is None:
            state = self.initial_state
            situation = Situation(state)
        else:
            state = situation.state
        state_ = self._do(action, state)
        situation_ = Situation(state_, action, situation)
        return situation_

    @cache
    def _do(self, action: Action, state: State) -> State:
        assert isinstance(state, State)
        if action.symbol == 'req':
            return State(state.world, state.k_relation, (action.arguments[0], *state.prioritized_goals))
        elif action.symbol == 'sense':
            sense: Literal = action.arguments[0]
            return State(state.world, FrozenRelation(frozenset({
                (elem[0], elem[1]) for elem in state.k_relation.relation_set if sense.sign == (abs(sense) in elem[1])
            })), state.prioritized_goals)
        else:
            raise NotImplementedError

    @cache
    def do_iter(self, actions: Sequence[Action], situation: Optional[Situation] = None) -> Situation:
        for action in actions:
            situation = self.do(action, situation)
        return situation

    @cache
    def know(self, formula: Formula, situation: Situation) -> bool:
        assert isinstance(situation, Situation)
        return all(formula.evaluate(self, situation_) for situation_ in self.k_relation.get_inverse(situation))

    @cache
    def _g_intersect(self, situation:Situation, level: int) -> Iterator[Path]:
        assert isinstance(situation, Situation)
        for s_ in situation.state.k_relation.get_inverse(situation.state.world):
            for p in self.all_paths(Situation(State(s_, situation.state.k_relation, situation.state.prioritized_goals))):
                if self._g_accessible(p, level, situation):
                    yield p
    @cache
    def _level_intention(self, formula: Formula, situation: Situation, level: int) -> bool:
        assert isinstance(situation, Situation)
        return all(formula.evaluate(self, path) for path in self._g_intersect(situation, level))

    @cache
    def _g_accessible(self, path: Path, level: int, situation: Situation):
        assert isinstance(situation, Situation)
        return len(situation.state.prioritized_goals) <= level or situation.state.prioritized_goals[level].evaluate(self, path)

    @cache
    def intention(self, formula: Formula, situation: Situation) -> bool:
        assert isinstance(situation, Situation)
        return all(self._level_intention(formula, situation, n) for n in range(len(situation.state.prioritized_goals)))

    def causes_directly(self) -> bool:
        pass

    def causes_indirectly(self) -> bool:
        pass

    def causes(self) -> bool:
        return self.causes_directly() or self.causes_indirectly()
#%%
@dataclass(order=True, frozen=True)
class Predicate(StateFormula):
    predicate: FluentLiteral

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return self.predicate in situation.state
#%%
@dataclass(order=True, frozen=True)
class Possible(StateFormula):
    action: Action

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return causal_setting.poss(self.action, situation)

#%%
@dataclass(order=True, frozen=True)
class After(StateFormula):
    action: Action
    formula: Formula

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return self.formula.evaluate(causal_setting, causal_setting.do(self.action, situation))
#%%
@dataclass(order=True, frozen=True)
class Negation(Formula):
    formula: Formula

    def evaluate(self, causal_setting: CausalSetting, elem: Union[State, Path]) -> bool:
        return not self.formula.evaluate(causal_setting, elem)

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return not self.formula.evaluate_situation(causal_setting, situation)

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return not self.formula.evaluate_path(causal_setting, path)

    def __neg__(self):
        return self.formula
#%%
@dataclass(order=True, frozen=True)
class Conjunction(Formula):
    left: Formula
    right: Formula

    def evaluate(self, causal_setting: CausalSetting, elem: Union[State, Path]) -> bool:
        return self.left.evaluate(causal_setting, elem) and self.right.evaluate(causal_setting, elem)

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return self.left.evaluate_situation(causal_setting, situation) and self.right.evaluate_situation(causal_setting,
                                                                                                         situation)

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return self.left.evaluate_path(causal_setting, path) and self.right.evaluate_path(causal_setting, path)

#%%
@dataclass(order=True, frozen=True)
class Disjunction(Formula):
    left: Formula
    right: Formula

    def evaluate(self, causal_setting: CausalSetting, elem: Union[State, Path]) -> bool:
        return self.left.evaluate(causal_setting, elem) or self.right.evaluate(causal_setting, elem)

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return self.left.evaluate_situation(causal_setting, situation) or self.right.evaluate_situation(causal_setting,
                                                                                                        situation)

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return self.left.evaluate_path(causal_setting, path) or self.right.evaluate_path(causal_setting, path)

#%%
@dataclass(order=True, frozen=True)
class Know(StateFormula):
    formula: Formula

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return causal_setting.know(self.formula, situation)
#%%
@dataclass(order=True, frozen=True)
class Intention(StateFormula):
    formula: Formula

    def evaluate_state(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return causal_setting.intention(self.formula, situation)

#%%
@dataclass(order=True, frozen=True)
class Next(PathFormula):
    formula: Formula

    def evaluate_path(self, causal_setting: _ForwardCausalSetting, path: Path) -> bool:
        for action in causal_setting.action_alphabet:
            if not causal_setting.poss(action, path.root):
                continue
            path_ = path.get_suffix(causal_setting.do(action, Situation(path.root)).state)
            if self.formula.evaluate_path(causal_setting, path_):
                return True
        return False


#%%
@dataclass(order=True, frozen=True)
class Until(PathFormula):
    holds_formula: Formula
    until_formula: Formula

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        for s_ in path.states:
            p_ = path.get_suffix(s_)
            if not self.until_formula.evaluate_path(causal_setting, p_):
                continue
            valid = True
            for s_star in path.states:
                p_star = path.get_suffix(s_star)
                if not self.holds_formula.evaluate_path(causal_setting, p_star):
                    valid = False
                    break
            if valid:
                return True
        return False
#%%
@dataclass(order=True, frozen=True)
class TrivialTrue(Formula):

    def evaluate(self, causal_setting: CausalSetting, elem: Union[State, Path]) -> bool:
        return True

    def evaluate_situation(self, causal_setting: CausalSetting, situation: Situation) -> bool:
        return True

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return True

#%%
@dataclass(order=True, frozen=True)
class Eventually(PathFormula):
    formula: Formula

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return Until(TrivialTrue(), self.formula).evaluate_path(causal_setting, path)
#%%
@dataclass(order=True, frozen=True)
class Before(PathFormula):
    before_formula: Formula
    after_formula: Formula

    def evaluate_path(self, causal_setting: CausalSetting, path: Path) -> bool:
        return Negation(Until(Negation(self.before_formula), self.after_formula)).evaluate_path(causal_setting, path)
#%%
Ls = Atom('Ls')
Ld = Atom('Ld')
L1 = Atom('L1')
L1_ = Atom("L1'")

takeOff_Ls = Atom('takeOff', (Ls,))
takeOff_Ld = Atom('takeOff', (Ld,))
takeOff_L1 = Atom('takeOff', (L1,))
takeOff_L1_ = Atom('takeOff',(L1_,))

flyTo_Ls_L1 = Atom('flyTo', (Ls,L1))
flyTo_Ls_L1_ = Atom('flyTo', (Ls,L1_))
flyTo_L1_Ld = Atom('flyTo', (L1,Ld))
flyTo_L1__Ld = Atom('flyTo', (L1_,Ld))

land_Ls = Atom('land',  (Ls ,))
land_Ld = Atom('land',  (Ld ,))
land_L1 = Atom('land',  (L1 ,))
land_L1_ = Atom('land', (L1_,))

atom_At_Ls = Atom('At', (Ls,))
atom_At_Ld = Atom('At', (Ld,))
atom_At_L1 = Atom('At', (L1,))
atom_At_L1_ = Atom('At', (L1_,))

At_Ls = Literal(atom_At_Ls)
At_Ld = Literal(atom_At_Ld)
At_L1 = Literal(atom_At_L1)
At_L1_ = Literal(atom_At_L1_)

a_Flying = Atom('Flying')

Flying = Literal(a_Flying)

atom_Vis_Ls = Atom('Vis', (Ls,))
atom_Vis_Ld = Atom('Vis', (Ld,))
atom_Vis_L1 = Atom('Vis', (L1,))
atom_Vis_L1_ = Atom('Vis', (L1_,))

Vis_Ls = Literal(atom_Vis_Ls)
Vis_Ld = Literal(atom_Vis_Ld)
Vis_L1 = Literal(atom_Vis_L1)
Vis_L1_ = Literal(atom_Vis_L1_)

atom_TStrom_Ls = Atom('TStrom', (Ls,))
atom_TStrom_Ld = Atom('TStrom', (Ld,))
atom_TStrom_L1 = Atom('TStrom', (L1,))
atom_TStrom_L1_ = Atom('TStrom', (L1_,))

TStrom_Ls = Literal(atom_TStrom_Ls)
TStrom_Ld = Literal(atom_TStrom_Ld)
TStrom_L1 = Literal(atom_TStrom_L1)
TStrom_L1_ = Literal(atom_TStrom_L1_)

sense_TStrom_Ls = Atom('sense', (TStrom_Ls,))
sense_TStrom_Ld = Atom('sense', (TStrom_Ld,))
sense_TStrom_L1 = Atom('sense', (TStrom_L1,))
sense_Tstrom_L1_ = Atom('sense', (TStrom_L1_,))

req_Eventually_Vis_L1_ = Atom('req', (Eventually(Predicate(Vis_L1_)),))

route = {
    Ls: frozenset({L1, L1_}),
    L1: frozenset({Ld}),
    L1_: frozenset({Ld}),
    Ld: frozenset()
}

drone_example_actions = {takeOff_Ls, takeOff_L1, takeOff_L1_, takeOff_Ld,
                         land_Ls, land_L1, land_L1_, land_Ld,
                         flyTo_Ls_L1, flyTo_Ls_L1_, flyTo_L1_Ld, flyTo_L1__Ld,
                         sense_TStrom_Ls, sense_TStrom_Ld, sense_TStrom_L1, sense_Tstrom_L1_,
                         req_Eventually_Vis_L1_}

drone_example_fluents = {
    atom_At_Ls, atom_At_L1, atom_At_L1_, atom_At_Ld,
    atom_Vis_Ls, atom_At_L1, atom_At_L1_, atom_At_Ld,
    atom_TStrom_Ls, atom_TStrom_L1, atom_TStrom_L1_, atom_TStrom_Ld
}
#%%
@dataclass(order=True, frozen=True)
class DroneExample(CausalSetting):

    def poss(self, action: Action, situation: Situation) -> bool:
        if action not in self.action_alphabet:
            return False
        if action.symbol in ('sense', 'req'):
            return super(DroneExample, self).poss(action, situation)
        l = action.arguments[0]
        atom_At_l = Atom('At', (l,))
        At_l = Literal(atom_At_l)
        if At_l not in situation.state:
            return False
        if action.symbol == 'takeOff':
            return -Flying in situation.state
        if action.symbol == 'land':
            return Flying in situation.state
        l_ = action.arguments[1]
        if action.symbol == 'flyTo':
            return Flying in situation.state and l_ in route[l]

        assert False, "Unexpected action {}".format(action)

    def _do(self, action: Action, state: State) -> State:
        if action.symbol == 'takeOff':
            def update(s):
                return (s - {-Flying}) | {Flying}
        elif action.symbol == 'land':
            def update(s):
                return (s - {Flying}) | {-Flying}
        elif action.symbol == 'flyTo':
            l = action.arguments[0]
            l_ = action.arguments[1]
            atom_At_l = Atom('At', l)
            atom_At_l_ = Atom('At', l_)
            At_l = Literal(atom_At_l)
            At_l_ = Literal(atom_At_l_)

            def update(s):
                return (s - {At_l}) | {-At_l, At_l_}
        else:
            return super(DroneExample, self)._do(action, state)
        return State(update(state.world), FrozenRelation(
            frozenset({(update(elem[0]), update(elem[1])) for elem in state.k_relation.relation_set})), state.prioritized_goals)




#%%
drone_example_initial_world = frozenset({
    Vis_Ls, At_Ls, TStrom_L1
})
print(drone_example_initial_world)
#%%
drone_example_k_states = {frozenset(Literal(atom) for atom in atoms) for atoms in
                          more_itertools.powerset(drone_example_fluents)}
len(drone_example_k_states)
#%%
drone_example_valid_k_states = {
    k_state for k_state in drone_example_k_states if
    At_Ls in k_state and
    At_L1 not in k_state and
    At_L1_ not in k_state and
    At_Ld not in k_state and
    Flying not in k_state and
    Vis_Ls in k_state and
    Vis_L1 not in k_state and
    Vis_L1_ not in k_state and
    Vis_Ld not in k_state and
    TStrom_Ls not in k_state and
    TStrom_L1_ not in k_state and
    TStrom_Ld not in k_state
}
len(drone_example_valid_k_states)
#%%
for k_state in drone_example_valid_k_states:
    print(k_state)
#%%
drone_example_k_relation_ground_coll = {
    (drone_example_initial_world, k_state) for k_state in drone_example_valid_k_states
}
drone_example_k_relation = FrozenRelation.from_ground_coll(drone_example_k_relation_ground_coll)
print(drone_example_k_relation)
#%%
drone_example_phi_0 = Eventually(Predicate(At_Ld))
drone_example_phi_1 = Before(Predicate(Vis_L1), Predicate(Vis_Ld))
#%%
drone_example_initial_state = State(frozenset(drone_example_initial_world), drone_example_k_relation, (drone_example_phi_0, drone_example_phi_1))
print(drone_example_initial_state)
#%%
-Flying
#%%
(-Flying) in drone_example_initial_state
#%%
Vis_Ls in drone_example_initial_state
#%%
drone_example = DroneExample(
    frozenset(drone_example_fluents),
    frozenset(drone_example_actions),
    drone_example_initial_state,
    drone_example_k_relation,
    4,
)
#%%
drone_example_s3 = drone_example.do_iter((takeOff_Ls, sense_TStrom_L1, req_Eventually_Vis_L1_))
print(len(drone_example_s3.parent.parent.parent.state.k_relation.relation_set))
drone_example_s3.parent.parent.parent
#%%
print(drone_example_s3.parent.parent.action)
print(len(drone_example_s3.parent.parent.state.k_relation.relation_set))
drone_example_s3.parent.parent
#%%
print(drone_example_s3.parent.action)
print(len(drone_example_s3.parent.state.k_relation.relation_set))
drone_example_s3.parent
#%%
print(drone_example_s3.action)
print(len(drone_example_s3.state.k_relation.relation_set))
drone_example_s3
#%%
#sum(1 for _ in drone_example.all_paths(Situation(drone_example_initial_state)))
#%%
print(drone_example.intention(drone_example_phi_0, drone_example_s3))
#%%
print(drone_example.intention(drone_example_phi_1, drone_example_s3))