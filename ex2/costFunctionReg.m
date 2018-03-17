function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
sigmoid = 1 ./ (1 + e .^ - (X * theta));
non_bias_theta = theta(2:size(theta, 1),1);
regularized_term = lambda * sum(non_bias_theta .^2);
J = 1 / m * (-y' * log(sigmoid) - (1 -y)' * log(1 -sigmoid) + regularized_term / 2);
grad = 1 / m * (X' *(sigmoid - y) + lambda * [0; non_bias_theta]);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
