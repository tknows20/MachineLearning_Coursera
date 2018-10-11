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

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');

% ---------Part 1--------------------------------------------------------------
% better implementation
for i=1:m,
 y_new(y(i),i)=1;
endfor

J_temp = (1/m) * sum ( sum ( (-y_new) .* log(h2') - (1-y_new) .* log(1-h2') ));

Theta1_new = Theta1(:,2:end);
Theta2_new = Theta2(:,2:end);
J_reg = (lambda/(2*m)) * (sum (Theta1_new(:).^2) + sum(Theta2_new(:).^2));
J = J_temp + J_reg;

% original implementation
%sum_J = 0;
%for k = 1: num_labels
%  y_k = (y == k);
%  J_temp = (-1 * y_k .* log(h2(:,k)) - ( (1-y_k) .* log(1-h2(:,k)) ) );
%  sum_J = sum_J + sum(J_temp(:));
%endfor
%J = sum_J/m;

% ---------Part 2--------------------------------------------------------------

Delta_1 = 0;
Delta_2 = 0;

for j = 1:m
	a_1 = [1; X(j, :)']; % Including Bias
	z_2 = Theta1 * a_1;
	a_2 = [1; sigmoid(z_2)]; % Including Bias

	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);

  y_temp = zeros(num_labels,1);
  y_temp(y(j)) = 1;
	d_3 = a_3 - y_temp;
  
	d_2 = (Theta2(:,2:end)' * d_3) .* sigmoidGradient(z_2);

	Delta_2 += (d_3 * a_2');
	Delta_1 += (d_2 * a_1');
endfor

Theta1_grad = (1 / m) * Delta_1;
Theta2_grad = (1 / m) * Delta_2;

Theta1_grad(:,2:end) += (lambda / m) * Theta1(:,2:end);
Theta2_grad(:,2:end) += (lambda / m) * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
