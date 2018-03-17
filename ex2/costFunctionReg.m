function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

predictions = sigmoid(X * theta);
non_bias_theta = theta(2:size(theta, 1),1);  % theta that will be regularized
regularized_term = (1 / m) * lambda * sum(non_bias_theta .^2);

% the regularized cost
J = 1 / m * (-y' * log(predictions) - (1 -y)' * log(1 -predictions)) + regularized_term;

% the regularized gradients
grad = 1 / m * (X' *(predictions - y) + lambda * [0; non_bias_theta]);

% =============================================================

end
