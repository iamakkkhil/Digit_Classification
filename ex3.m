%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, nu  -3.25328   0.00000   0.00000  -0.00003   0.00040  -0.00092  -0.01640  -0.00912  -0.01320  -0.00159
  -4.96857   0.00000   0.00000  -0.00001  -0.00014   0.00375   0.01540  -0.02172  -0.00632  -0.00062
  -2.34534   0.00000   0.00000  -0.00001  -0.00002   0.00134  -0.00094   0.00101   0.05175   0.03930
   0.61461   0.00000   0.00000  -0.00000   0.00008  -0.00084  -0.00749  -0.00317  -0.01529  -0.00919
  -3.62683   0.00000   0.00000  -0.00000   0.00002  -0.00013  -0.00025  -0.00144  -0.00106  -0.00020
  -2.07763   0.00000   0.00000  -0.00003   0.00018   0.00077   0.00462   0.04465   0.00254  -0.00222
  -8.29939   0.00000   0.00000  -0.00001   0.00005   0.00009  -0.00012  -0.00776  -0.00655   0.00064
  -4.36086   0.00000   0.00000  -0.00000   0.00002   0.00024  -0.00443  -0.00340  -0.00094  -0.01197
  -6.18264   0.00000   0.00000  -0.00000  -0.00001   0.00013   0.00092   0.00004  -0.00050  -0.00087

 Columns 11 through 20:

   0.00047   0.00007   0.00036   0.00162   0.00142m_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

