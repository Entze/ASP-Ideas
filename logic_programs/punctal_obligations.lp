
obligationtype(punctual).


% A15: "punctual obligation d_holdsAt"
d_holdsAt(obligation(punctual, Ts, X), Ts) :-
  happens(trigger(obligation(punctual, Ts, X), N), Tt),
  Ts = Tt + N, N >= 0.

% A16: "punctual obligation d_terminates"
d_terminates(obligation(punctual, Ts, X), Ts) :-
  happens(trigger(obligation(punctual, Ts, X), N), Tt),
  Ts = Tt + N, N >= 0.

% A17: "punctual obligation violation"
happens(violation(obligation(punctual, Ts, X)), Tv) :-
  d_holdsAt(obligation(punctual, Ts, X), Ts),
  not happens(X, Ts), not holdsAt(X, Ts), Tv = Ts.