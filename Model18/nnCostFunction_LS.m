function [J grad] = nnCostFunction(nn_params, ...
                                   inp_size, ...
                                   hidden_size, ...
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

% 3 layers
Theta1f = reshape(nn_params(1:hidden_size(1) * (inp_size + 1)), ...
                 hidden_size(1), (inp_size + 1));

Theta2f = reshape(nn_params(1 + (hidden_size(1) * (inp_size + 1)):(hidden_size(1) * (inp_size + 1)) + (hidden_size(2) * (hidden_size(1) + 1))), ...
                 hidden_size(2), (hidden_size(1) + 1));
             
Theta3f = reshape(nn_params((hidden_size(1) * (inp_size + 1)) + (hidden_size(2) * (hidden_size(1) + 1)) + 1:end), ...
                 num_labels, (hidden_size(2) + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1f));
Theta2_grad = zeros(size(Theta2f));
Theta3_grad = zeros(size(Theta3f));

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
 
 X = [ones(m,1) X];

 a2 = [ones(m,1), relu(X*Theta1f')];
 a3 = [ones(m,1), relu(a2*Theta2f')];
 a4 = relu(a3*Theta3f');
   
 J = 0.5*sum(sum((a4-y).^2))/m + (lambda*(sum((Theta1f(:,2:end).^2),'all') + sum((Theta2f(:,2:end).^2),'all') + sum((Theta3f(:,3:end).^2),'all')))/(2*m);
  
% -------------------------------------------------------------
Delta1 = zeros(size(Theta1f));
Delta2 = zeros(size(Theta2f));
Delta3 = zeros(size(Theta3f));
for t = 1:m
    a1 = X(t,:)';
    z1 = Theta1f*a1;
    a2 = [1; relu(z1)];
    z2 = Theta2f*a2;
    a3 = [1; relu(z2)];
    z3 = Theta3f*a3;
    a4 = relu(z3);
    del4 = a4 - y(t,:)';
    del3 = (Theta3f'*del4);
    del3 = del3(2:end);
    del2 = (Theta2f'*del3);
    del2 = del2(2:end);
    Delta1 = Delta1 + del2*a1';
    Delta2 = Delta2 + del3*a2';
    Delta3 = Delta3 + del4*a3';
end
% =========================================================================
Theta1_grad = Delta1./m + (lambda/m).*[zeros(hidden_size(1),1), Theta1f(:,2:end)];
Theta2_grad = Delta2./m + (lambda/m).*[zeros(hidden_size(2),1), Theta2f(:,2:end)];
Theta3_grad = Delta3./m + (lambda/m).*[zeros(num_labels,1), Theta3f(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];


end
