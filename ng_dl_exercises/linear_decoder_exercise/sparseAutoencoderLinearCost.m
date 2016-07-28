function [cost,grad] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------    

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

m = size(data,2);
x = data;
z2 = W1 * x + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = z3;
h = a3;   

rho = sparsityParam;
rho_hat = mean(a2,2);

cost = sum(sum((h-x).^2))/(2*m) + ...
    lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2))) + ...
    beta * sum(rho*log(rho./rho_hat)+(1-rho)*log((1-rho)./(1-rho_hat)));
    
separsity_dalta = -rho./rho_hat+(1-rho)./(1-rho_hat);

delta3 = -(x-h);
delta2 = (W2'*delta3 + repmat(beta*separsity_dalta,1,m)).*a2.*(1-a2);

Delta_W2 = delta3*a2';
Delta_W1 = delta2*x'; 
Delta_b2 = sum(delta3,2);
Delta_b1 = sum(delta2,2); 

W2grad = Delta_W2/m + lambda*W2; 
W1grad = Delta_W1/m + lambda*W1;
b2grad = Delta_b2/m; 
b1grad = Delta_b1/m;  

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];                       

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
