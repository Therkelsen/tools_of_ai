clc; close; clear all;

w1 = 0.5;
w2 = 0.5;
w3 = 0.5;
w4 = 0.5;
w5 = 0.5;
w6 = 0.5;

b1 = 0.1;
b2 = 0.1;
b3 = 0.1;

lr = 0.3;

x1 = 0;
x2 = 0;
ytarg = 0;

h1 = sig(w1*x1+w4*x2+b1)
h2 = sig(w3*x1+w2*x2+b2)

y = sig(w5*h1+w6*h2+b3)

E = 1/2*(ytarg-y)^2

delta3 = (ytarg - y) * y*(1-y)
delta1 = delta3 * w5 * h1*(1-h1)
delta2 = delta3 * w6 * h2*(1-h2)

w5 = w5 - lr * delta3 * h1
w6 = w6 - lr * delta3 * h2

w1 = w1 - lr * delta1 * x1
w2 = w2 - lr * delta1 * x2

w3 = w3 - lr * delta2 * x1
w4 = w4 - lr * delta2 * x2

b3 = b3 - lr * delta3
b1 = b1 - lr * delta1
b2 = b2 - lr * delta2

function sig = sig(x)
    sig = 1/(1+exp(-x));
end