function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
fprintf('--------------------------------------------------------------------------------\n');
fprintf('START SEARCHING BEST VALUES FOR C AND SIGMA\n');

error_min = inf;
values = [0.01 0.03 0.1 0.3 1 3 10 30];

for i = values
  for j = values                                                   
    fprintf(['\nTry on cross validation set: C   = %f\n' ...
             '                           Sigma = %f\n'], i, j);
    model = svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
    error = mean(double(svmPredict(model, Xval) ~= yval));
    fprintf('Prediction error: %f\n', error);

    if (error == error_min)
          fprintf('Same best result than before\n');
    end
    
    if (error < error_min)
      C = i;
      sigma = j;
      error_min = error;
      fprintf('Smaller error founded, now C = %f and Sigma = %f\n', C, sigma);
    end

    if (error > error_min)
        fprintf('No better results founded. Next\n');
    end

    fprintf('-------------------------------------------------------------------------------\n');

  end
end

fprintf('Proccess finished\n Best value [C, sigma] = [%f %f] with Prediction Error = %f\n\n', C, sigma, error_min);
fprintf('--------------------------------------------------------------------------------\n');

% =========================================================================

end
