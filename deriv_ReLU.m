function g = deriv_ReLU(z)
z(z>0)=1;
z(z<0)=0;
g = z;
end