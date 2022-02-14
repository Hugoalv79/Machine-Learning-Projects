function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda)
%   COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end),num_users, num_features);
            
% You need to return the following values correctly
J = 0;
XGrad = zeros(size(X));
ThetaGrad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
%{
Instructions: Compute the cost function and gradient for collaborative
               filtering. Concretely, you should first implement the cost
               function (without regularization) and make sure it is
               matches our costs. After that, you should implement the 
               gradient and use the checkCostFunction routine to check
               that the gradient is correct. Finally, you should implement
               regularization.

 Notes: X - num_movies  x num_features matrix of movie features
        Theta - num_users  x num_features matrix of user features
        Y - num_movies x num_users matrix of user ratings of movies
        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
            i-th movie was rated by the j-th user

 You should set the following variables correctly:

        XGrad - num_movies x num_features matrix, containing the 
                 partial derivatives w.r.t. to each element of X
        ThetaGrad - num_users x num_features matrix, containing the 
                     partial derivatives w.r.t. to each element of Theta
%}

%{ 
To come up with a vectorized implementation, the following tip might be helpful: 
You can use the R matrix to set selected entries to 0. 
For example, R .* M will do an element-wise multiplication between M and R; 
since R only has elements with values either 0 or 1, this has the effect of setting 
the elements of M to 0 only when the corresponding value in R is 0. 
Hence, sum(sum(R.*M)) is the sum of all the elements of M for which the corresponding 
element in R equals 1.
%}

% Compute collaborative filtering cost function
error = ((X * Theta') - Y);
J = 1/2 * sum(sum(R .* (error.^2)));

% Add regularized gradients terms

RegTermXGrad = lambda * X;
RegTermThetaGrad = lambda * Theta;
		
% Compute collaborative filtering gradients
XGrad = ((X * Theta' - Y) .* R) * Theta;
XGrad = XGrad + RegTermXGrad;


ThetaGrad = ((X * Theta' - Y) .* R)' * X;
ThetaGrad = ThetaGrad + RegTermThetaGrad;

% Compute regularized cost function

RegTermJ = (lambda/2) * (sum(sum(Theta.^2)) + sum(sum(X.^2))); 
J = J + RegTermJ;

% =============================================================

grad = [XGrad(:); ThetaGrad(:)];
end
