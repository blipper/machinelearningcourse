function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h=@(theta, X) sigmoid(X * theta);
%fprintf(' %f \n', log(h(theta,X)));

%J = (1/m) * sum(((-1*y) .* log(h(theta,X)) - ((1-y) .* log(h(theta, 1-X)))));
J = (-1/m) * sum((log(h(theta,X)))'*y + (log(1-h(theta, X)))'*(1-y) ) + (lambda/(2*m)) * sum((theta.^2)(2:end));

grad = (1/m) * (X' * (h(theta, X) - y)) + [0;(lambda/m)*theta(2:end)];



% =============================================================

end