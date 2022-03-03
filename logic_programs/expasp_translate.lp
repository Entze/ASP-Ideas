a :- k, not b.
k :- e, not b.
c :- a, b.

b :- not a.
c :- k.
f :- e, not k, not c.
#external e.