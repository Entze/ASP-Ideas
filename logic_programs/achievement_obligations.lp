
obligationstype(achievement).

% A19: "achievement obligation d_holdsAt Start"
%!trace_rule {"A19: % has to be achieved since % as it was triggered at %.", X, Time_Start, Time_Trigger}
d_holdsAt(obligation(achievement, Time_Start, X), Time_Start) :-
  happens(trigger(obligation(achievement, Time_Start, X), Delay), Time_Trigger), Time_Start = Time_Trigger + Delay, Delay >= 0.

% A20: "achievement obligation d_holdsAt inertia"
%!trace_rule {"A20: % has to be achieved since % and at % as it was not clipped yet.", X, Time_Start, Time_During}
d_holdsAt(obligation(achievement, Time_Start, X), Time_During) :-
  time(Time_During),
  d_holdsAt(obligation(achievement, Time_Start, X), Time_Start), not d_clipped(Time_Start, obligation(achievement, Time_Start, X), Time_During),
  Time_Start <= Time_During.

% A21: "achievement obligation d_termination by event"
%!trace_rule {"A21a."}
d_terminates(E, obligation(achievement, Time_Start, X), Time_During) :-
  happens(E, Time_During), d_holdsAt(obligation(achievement, Time_Start, X), Time_During),
  happens(X, Time_During), fulfillTerminable(obligation(achievement, Time_Start, X)), Time_Start <= Time_During.

%!trace_rule {"A21b."}
d_terminates(E, obligation(achievement, Time_Start, X), Time_During) :-
  happens(E, Time_During), d_holdsAt(obligation(achievement, Time_Start, X), Time_During),
  holdsAt(X, Time_During), fulfillTerminable(obligation(achievement, Time_Start, X)), Time_Start <= Time_During.

% A22: "achievement obligation d_terminates on deadline"
%!trace_rule {"A22b."}
d_terminates(deadline(obligation(achievement, Time_Start, X), Time_Deadline), obligation(achievement, Time_Start, X), Time_Deadline) :-
  happens(deadline(obligation(achievement, Time_Start, X)), Time_Deadline), Time_Start <= Time_Deadline.

% A23: "achievement obligation d_terminates preemptive"
%!trace_rule {"A23a."}
d_terminates(E, obligation(achievement, Time_Start, X), Time_End) :-
  happens(E, Time_End), d_holdsAt(obligation(achievement, Time_Start, X), Time_Start),
  time(T), happens(X, T), fulfillTerminable(obligation(achievement, Time_Start, X)), Time_End = Time_Start + 1, T < Time_Start.

%!trace_rule {"A23b."}
d_terminates(E, obligation(achievement, Time_Start, X), Time_End) :-
  happens(E, Time_End), d_holdsAt(obligation(achievement, Time_Start, X), Time_Start),
  time(T), holdsAt(X, T), fulfillTerminable(obligation(achievement, Time_Start, X)), Time_End = Time_Start + 1, T < Time_Start.

% A24: "achievement obligation violation"
%!trace_rule {"A24."}
happens(violation(obligation(achievement, Time_Start, X)), Time_Violation) :-
  d_holdsAt(obligation(achievement, Time_Start, X), Time_End),
  happens(deadline(obligation(achievement, Time_Start, X)), Time_End),
  not happens(X, Time_End), not holdsAt(X, Time_End),
  violationTerminable(obligation(achievement, Time_Start, X)), Time_Violation = Time_End.