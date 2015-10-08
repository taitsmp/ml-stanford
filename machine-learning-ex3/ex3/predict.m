function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

size(Theta1)
size(Theta2)

fprintf('Printed sizes.\n');

X = [ones(m, 1) X];

Z1 = Theta1 * X';
A1 = sigmoid(Z1');
a1c = size(A1,1);
A1 = [ones(a1c, 1) A1];

fprintf('A1 size\n');
size(A1)

Z2 = Theta2 * A1'
HT = sigmoid(Z2');

fprintf('HT size\n');
size(HT)

[P, p] = max(HT, [], 2);


% =========================================================================


end
