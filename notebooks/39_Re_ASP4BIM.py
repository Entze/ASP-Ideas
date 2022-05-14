from typing import Sequence, Optional, Dict

import clingo.ast
from clingo import PropagateInit, PropagateControl
from clingo.ast import ProgramBuilder
from shapely.geometry import Polygon

db = {
    "square1": Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
    "square2": Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
}
print(db)

data = """
type(polygon).

product(polygon, "square1").
product(polygon, "square2").
"""

program = """
&intersect{ "square1"; "square2" } = intersect(polygon, "intersect_1_2").
&union{ "square1"; "square2" } = union(polygon, "union_1_2").
"""

theory = """
#theory spatial {
	constant  {};
	spatial_term {};
	&union/0 : spatial_term, {=}, constant, any;
	&intersect/0 : spatial_term, {=}, constant, any
}.
"""


def unwrap(theory_term: clingo.TheoryTerm):
    if theory_term.type is clingo.TheoryTermType.Symbol:
        return theory_term.name[1:-1]
    elif theory_term.type is clingo.TheoryTermType.Number:
        return theory_term.number
    elif theory_term.type is clingo.TheoryTermType.Function:
        return str(theory_term)
    arguments = (unwrap(argument) for argument in theory_term.arguments)
    if theory_term.type is clingo.TheoryTermType.Set:
        return set(arguments)
    elif theory_term.type is clingo.TheoryTermType.Tuple:
        return tuple(arguments)
    elif theory_term.type is clingo.TheoryTermType.List:
        return list(arguments)
    else:
        assert False, "Unknown TheoryTermType {} of TheoryTerm {}.".format(theory_term.type, theory_term)


class SpatialTransformer(clingo.ast.Transformer):
    pass


class SpatialPropagator(clingo.Propagator):

    def __init__(self, database: Optional[Dict[str, Polygon]] = None):
        self._l2t = {}  # literal -> (op, geoms, assign)
        self._a2l = {}  # assign -> literal
        self._l2s = {}  # literal -> solver_literal
        self._s2l = {}  # solver_literal -> literal
        self._qrst = QRST(database)

    def init(self, init: PropagateInit) -> None:
        for atom in init.theory_atoms:
            term = atom.term
            op = term.name
            geoms = sorted(unwrap(element.terms[0]) for element in atom.elements)
            assign = unwrap(atom.guard[1])
            program_literal = atom.literal
            solver_literal = init.solver_literal(program_literal)
            self._l2t[program_literal] = (op, geoms, assign)
            self._l2s[program_literal] = solver_literal
            self._s2l[solver_literal] = program_literal
            self._a2l[assign] = program_literal
            init.add_watch(solver_literal)
            if init.assignment.is_true(solver_literal):
                self.propagate_assignment(op, geoms, assign)

    def propagate(self, control: PropagateControl, changes: Sequence[int]) -> None:
        for change in changes:
            if change in self._s2l:
                program_literal = self._s2l[change]
                op, geoms, assign = self._l2t[program_literal]
                self.propagate_assignment(op, geoms, assign)

    def propagate_assignment(self, op, geoms, assign):
        self._qrst.evaluate(op, geoms, assign)


class QRST:

    def __init__(self, database: Optional[Dict[str, Polygon]] = None):
        self._db: Dict[str, Polygon] = database or {}
        self._assignments: Dict[str, str] = {}

    def evaluate(self, operation: str, geoms: str, assign: str) -> None:
        geom_id: str = "{}({})".format(operation, ','.join(sorted(geoms)))
        if geom_id not in self._db:
            if operation == 'intersect':
                self.intersect(geom_id, geoms)

        self._assignments[assign] = geom_id

    def intersect(self, geom_id: str, geoms: Sequence[str]) -> None:
        polys = list(self._db[geom] for geom in geoms)
        new_poly = polys[0]
        i = 1
        while i < len(polys):
            new_poly = new_poly.intersection(polys[i])
            i += 1
        self._db[geom_id] = new_poly


ctl = clingo.Control()
ctl.configuration.solve.models = 0

propagator = SpatialPropagator(db)
transformer = SpatialTransformer()

ctl.register_propagator(propagator)

ctl.add('base', [], theory)
print(theory)
ctl.add('base', [], data)
print(data)
stmts = []
clingo.ast.parse_string(program, lambda stm: stmts.append(transformer.visit(stm)))
with ProgramBuilder(ctl) as builder:
    for stm in stmts:
        print(stm)
        builder.add(stm)

ctl.ground([('base', ())])

with ctl.solve(yield_=True) as solve_handle:
    models = []
    for model in solve_handle:
        symbols = sorted(model.symbols(shown=True, theory=True))
        models.append(model)
        print("Answer {}:".format(model.number), end=' ')
        print("{",
              '\n'.join(map(str, symbols)), "}", sep='\n')
    solve_result = solve_handle.get()
    print(solve_result, end='')
    if models:
        print(" {}{}".format(len(models), '' if solve_result.exhausted else '+'))

print(propagator._l2t)

print(db)
