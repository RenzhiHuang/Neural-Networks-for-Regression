function [error,y_hat] = ...
    neural_networks_sqerror(testdata,testlabel,w1,b1,w2,b2)
d = size(testdata,2);
m = size(testdata,1);
z = w1*testdata'+repmat(b1,1,m);% d1*m
a1 = sigmoid(z); % d1*m
%a1 = ReLU(z); % d1*m
y_hat = w2 * a1+repmat(b2,1,m);% 1*m
error = 1/m*sum((y_hat'-testlabel).^2);
end