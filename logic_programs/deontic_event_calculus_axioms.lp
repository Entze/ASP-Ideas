
% A13: "d_holdsAt final negation"
%!trace_rule {"A13."}
-d_holdsAt(P, T + 1) :-
  d_terminates(X, P, T).

%!trace_rule {"Expanding d_terminates."}
d_terminates(trigger(X, Delay), P, Delay, T) :- d_terminates(trigger(X, Delay), P, T).

% A14: "-d_holdsAt inertia axiom"
%!trace_rule {"A14."}
-d_holdsAt(P, T3) :-
  time(T3),
  -d_holdsAt(P, T1), -happens(trigger(P, Delay), T2),
  T1 < T3, T1 <= T2 + Delay, T2 + Delay <= T3.

% "d_clipped definition"
d_clipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(trigger(X, Delay), T), T1 <= T, T < T2, d_terminates(X, P, T).

%!trace {"% d_holdsAt %", P, T} d_holdsAt(P, T).
%!trace {"% d_terminates % at %", X, P, T} d_terminates(X, P, T).
%!trace {"% is terminable on violation", X} terminableViolation(X).
%!trace {"% is terminable on fulfillment", X} fulfillTerminable(X).
