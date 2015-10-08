function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1) X]; % add column of ones.
J = 0;

%forward prop
% X => m x n
% Theta1 => hl x n
% Theta2 => nl x hl 
A2 = sigmoid(X*Theta1');  % m x hl
A2 = [ones(m,1) A2];
Ht = sigmoid(A2*Theta2'); % m x nl

for k = 1:num_labels
  yb = y == k; % m x 1
	ht = Ht(:, k); % m x 1
	J = J + (1/m) * (-yb'*log(ht) - (1-yb')*log(1-ht));
end

Theta1nb = Theta1(:, 2:end);
Theta2nb = Theta2(:, 2:end);
J = J + (lambda / (2*m)) * (sum(sum(Theta1nb.^2)) + sum(sum(Theta2nb.^2)));

%calc gradients
D1 = zeros(hidden_layer_size, input_layer_size+1);
D2 = zeros(num_labels, hidden_layer_size+1);
for t = 1:m
	%forward prop
	a1 = X(t,:)'; %already added a column of ones to X -> n x 1
	z2 = Theta1*a1; %hl x 1
	a2 = sigmoid(z2); % hl x 1
	a2 = [1; a2]; % hl x 1
	z3 = Theta2*a2; % nl x 1
	a3 = sigmoid(z3);

	%back prop
  if num_labels == 10
		tn = [1:9, 0]';
	else
		tn = [1:num_labels]';
	end
	ym = tn == y(t);
	d3 = a3 - ym; % nl x 1
	td3 = (Theta2'*d3);
	d2 = td3(2:end).*sigmoidGradient(z2);  % hl x 1
	D2 = D2 + d3*a2'; % nl x hl    
	D1 = D1 + d2*a1'; % hl x n
end

Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;	
reg1 = (lambda/m)*Theta1;
reg1(:, 1) = zeros(size(Theta1,1),1);
reg2 = (lambda/m)*Theta2;
reg2(:, 1) = zeros(size(Theta2,1),1);
Theta1_grad = Theta1_grad + reg1;
Theta2_grad = Theta2_grad + reg2;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
