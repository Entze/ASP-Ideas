
obligationtype(persistent).

% A18: "persistent obligation d_holdsAt"
d_holdsAt(obligation(persistent, Time_Start, X), Time_During) :-
  time(Time_During),
  happens(trigger(obligation(persistent, Time_Start, X), Delay), Time_Trigger),
  not d_clipped(Time_Start, obligation(persistent, Time_Start, X), Time_During),
  d_terminates(trigger(obligation(persistent, Time_Start, X), Delay), Time_End),
  Time_Start = Time_Trigger + Delay, Time_End > Time_Start, Time_Start <= Time_During, Time_During <= Time_End, Delay >= 0.