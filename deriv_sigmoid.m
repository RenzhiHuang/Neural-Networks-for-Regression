function g = deriv_sigmoid(z)
g = sigmoid(z).*(1-sigmoid(z));
end