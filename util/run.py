import datetime
import os
import time

import clingo
from clingo import SymbolType, Symbol


def solve(program, filter_symbols=None, nr_of_models: int = 0, report=True):
    ctl = clingo.Control(["--models", str(nr_of_models)])
    if isinstance(program, str):
        if os.path.isfile(program):
            # program is a path
            ctl.load(program)
        else:
            ctl.add("base", [], program)

    ctl.ground([("base", [])])

    with ctl.solve(yield_=True, async_=True) as solver:

        done = False
        i = 0
        calc_time = 0.
        models = []
        answer_sets = []

        while not done:
            stime = time.monotonic()
            solver.resume()
            _ = solver.wait()
            delta_time = time.monotonic() - stime
            calc_time += delta_time
            model = solver.model()
            done = model is None
            if not done:
                i += 1
                models.append(model)
                symbols = model.symbols(shown=True)
                answer_set = set()
                if filter_symbols is not None:
                    if isinstance(filter_symbols, list):
                        for filter_symbol in filter_symbols:
                            for symbol in symbols:
                                if compare_symbols(filter_symbol, symbol):
                                    answer_set.add(symbol)
                    else:
                        for symbol in symbols:
                            if compare_symbols(filter_symbols, symbol):
                                answer_set.add(symbol)
                else:
                    answer_set = set(symbols)

                answer_sets.append(answer_set)
                if report:
                    if delta_time > 0.01:
                        time_out = " in {}".format(datetime.timedelta(seconds=calc_time))
                    else:
                        time_out = ""
                    cost = model.cost
                    optimal = model.optimality_proven
                    if cost:
                        cost_out = "@{}{}".format(cost, " optimal" if optimal else "")
                    else:
                        cost_out = ""

                    print(
                        "Answer {:2d}: {}{}{}{}.{}".format(model.number, "{ ", model, " }",
                                                           cost_out, time_out))
        result = solver.get()
        if result.unknown:
            status_out = "UNKWN"
        elif result.satisfiable:
            status_out = "SAT"
        else:
            assert result.unsatisfiable
            status_out = "UNSAT"

        if result.interrupted:
            status_out += " (INTERRUPTED)"

        nr_of_solutions_out = str(i)
        if not result.exhausted:
            nr_of_solutions_out += "+"

        if calc_time > 0.1:
            time_out = datetime.timedelta(seconds=calc_time)
        else:
            time_out = ""
        if report:
            print(status_out, nr_of_solutions_out, time_out)
        return answer_sets, models


def compare_symbols(filter_symbol, symbol: Symbol):
    symbol_name = None
    symbol_type = None
    predicate_nr_of_args = None
    if isinstance(filter_symbol, Symbol):
        return filter_symbol == symbol
    elif isinstance(filter_symbol, str):
        if filter_symbol.endswith("."):
            symbol_name, predicate_nr_of_args = filter_symbol.split('/')
            predicate_nr_of_args = int(predicate_nr_of_args[0:-2])
        else:
            symbol_name = filter_symbol
    elif isinstance(filter_symbol, SymbolType):
        symbol_type = filter_symbol
    elif isinstance(filter_symbol, tuple):
        for e in filter_symbol:
            if isinstance(e, str):
                symbol_name = e
            elif isinstance(e, SymbolType):
                symbol_type = e
            elif isinstance(e, int):
                predicate_nr_of_args = e
    return (symbol_name is not None or symbol_type is not None or predicate_nr_of_args is not None) and (
            symbol_name is None or symbol.name == symbol_name) and (
                       symbol_type is None or symbol.type == symbol_type) and (
                   predicate_nr_of_args is None or predicate_nr_of_args == len(symbol.arguments))
