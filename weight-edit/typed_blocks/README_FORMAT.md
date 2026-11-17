Example: hand-typed weights for ONE topic of block "sci" (topic 0):
listen {"moon": 3, "far": 1}, reply "The Moon is 384,000 km away."

# topic sci.t0
E moon tok:moon 1
E far tok:far 1
B moon -1000000000
B far -1000000000
V tok:moon sci.t0 3
V tok:far sci.t0 1
O sci.t0 top:sci.t0 100
W1 top:sci.t0 n:sci.t0 1
W2 n:sci.t0 out:sci.t0.w0 1
E sci.t0.w0 tok:sci.t0.w0 1
E sci.t0.w1 tok:sci.t0.w1 1
E sci.t0.w2 tok:sci.t0.w2 1
E sci.t0.w3 tok:sci.t0.w3 1
E sci.t0.w4 tok:sci.t0.w4 1
E sci.t0.w5 tok:sci.t0.w5 1
S sci.t0.w0 The
S sci.t0.w1 Moon
S sci.t0.w2 is
S sci.t0.w3 384,000
S sci.t0.w4 km
S sci.t0.w5 away.
W1 tok:sci.t0.w0 n:sci.t0.w0 1
W2 n:sci.t0.w0 out:sci.t0.w1 1000
W1 tok:sci.t0.w1 n:sci.t0.w1 1
W2 n:sci.t0.w1 out:sci.t0.w2 1000
W1 tok:sci.t0.w2 n:sci.t0.w2 1
W2 n:sci.t0.w2 out:sci.t0.w3 1000
W1 tok:sci.t0.w3 n:sci.t0.w3 1
W2 n:sci.t0.w3 out:sci.t0.w4 1000
W1 tok:sci.t0.w4 n:sci.t0.w4 1
W2 n:sci.t0.w4 out:sci.t0.w5 1000
W1 tok:sci.t0.w5 n:sci.t0.w5 1
W2 n:sci.t0.w5 out:<end> 1000
U out:sci.t0.w0 sci.t0.w0 1
U out:sci.t0.w1 sci.t0.w1 1
U out:sci.t0.w2 sci.t0.w2 1
U out:sci.t0.w3 sci.t0.w3 1
U out:sci.t0.w4 sci.t0.w4 1
U out:sci.t0.w5 sci.t0.w5 1
