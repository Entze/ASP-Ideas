
obligationtype(punctual).


% A15: "punctual obligation d_holdsAt"
d_holdsAt(obligation(punctual, Time_Start, X), Time_Start) :-
  happens(trigger(obligation(punctual, Time_Start, X), Delay), Time_Trigger),
  Time_Start = Time_Trigger + Delay, Delay >= 0.

% A16: "punctual obligation d_terminates"
d_terminates(trigger(obligation(punctual, Time_Start, X), Delay), obligation(punctual, Time_Start, X), Time_Start) :-
  happens(trigger(obligation(punctual, Time_Start, X), Delay), Time_Trigger),
  Time_Start = Time_Trigger + Delay, Delay >= 0.

% A17: "punctual obligation violation"
happens(violation(obligation(punctual, Time_Start, X)), Time_Violation) :-
  d_holdsAt(obligation(punctual, Time_Start, X), Time_Start),
  not happens(X, Time_Start), not holdsAt(X, Time_Start), Time_Violation = Time_Start.