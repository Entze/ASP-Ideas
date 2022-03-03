
% Page 165 (M. Hashmi's Thesis)

%!trace_rule {"% is the starttime.", T}
starttime(T) :- time(T), not time(T-1).
%!trace_rule {"% is the endtime.", T}
endtime(T) :- time(T), not time(T+1).

% A1: "holdsAt inertia axiom"
%!trace_rule {"A1: % holds at %. It held at % and it was not clipped in [%,%].", P, T2, T1, T1, T2}
holdsAt(P, T2) :-
  time(T1), time(T2),
  holdsAt(P,T1), T1 < T2, not clipped(T1, P, T2).

% A2:  "holdsAt braking axiom"
%!trace_rule {"A2: % doesn't hold at %. % happened at % and this terminated it, and is was not declipped in [%,%].", P, T2, X, T1, T1, T1}
-holdsAt(P, T2) :-
  time(T1), time(T2),
  happens(X,T1), terminates(X,P,T1), T1 < T2, not declipped(T1, P, T2).

% A3: "holdsAt momentum axiom"
%!trace_rule {"A3: % holds at %. % happened at % and this initiated it, and it was not clipped in [%,%].",P, T2, X, T1, T1, T2}
holdsAt(P, T2) :-
  time(T1), time(T2),
  happens(X, T1), initiates(X,P,T1), T1 < T2, not clipped(T1, P, T2).

% A4: "-holdsAt inertia axiom"
%!trace_rule {"A4: % doesn't hold at %. It didn't hold at % and it was not declipped in [%,%].", P, T2, T1, T1, T2}
-holdsAt(P, T2) :-
  time(T1), time(T2),
  -holdsAt(P, T1), T1 < T2, not declipped(T1, P, T2).

% A5: "clipped definition"
%!trace_rule {"A5: % is clipped in [%,%], as % happened at % which terminated it." P, T1, T2, X, T}
clipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(X,T), T1 <= T, T < T2, terminates(X, P, T).

% A6: "declipped definition"
%!trace_rule {"A6: % is declipped in [%,%], as % happened at % which initiated it." P, T1, T2, X, T}
declipped(T1, P, T2) :-
  time(T1), time(T2),
  happens(X, T), T1 <= T, T < T2, initiates(X, P, T).

% "intializers"

%!trace_rule {"% holds at % as it is initially true.", P, ST}
holdsAt(P, ST) :-
  starttime(ST), initiallyTrue(P).

%!trace_rule {"% doesn't hold at % as it is initially false.", P, ST}
-holdsAt(P, ST) :-
  starttime(ST), initiallyFalse(P).

%!trace {"% holdsAt %", P, T} holdsAt(P, T).
%!trace {"% happened at %", E, T} happens(E, T).
%!trace {"% initiates % at %", E, P, T} initiates(E, P, T).
%!trace {"% terminates % at %", E, P, T} terminates(E, P, T).
