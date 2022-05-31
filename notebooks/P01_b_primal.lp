%!trace_rule {"a :- k, not b."}
a :- k, not b.
%!trace_rule {"c :- a, b."}
c :- a, b.
%!trace_rule {"c :- k."}
c :- k.
%!trace_rule {"k :- e, not b."}
k :- e, not b.

%!trace_rule {"e."}
e.

%!trace_rule {"b :- not a."}
b :- not a.
%!trace_rule {"f :- e, not k, not c."}
f :- e, not k, not c.

%!show_trace a.
%!show_trace c.
%!show_trace k.

%!show_trace e.

%!show_trace b.
%!show_trace f.