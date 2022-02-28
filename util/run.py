import datetime
import os
import time

import clingo


def solve(program, nr_of_models: int = 0):
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

                print("Answer {:2d}: {}{}{}{}.{}".format(model.number, "{ ", model, " }", cost_out, time_out))
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

        print(status_out, nr_of_solutions_out, time_out)
