function [softmaxModel] = softmaxTrain(inputSize, numClasses, lambda, inputData, labels, options)
%softmaxTrain Train a softmax model with the given parameters on the given
% data. Returns softmaxOptTheta, a vector containing the trained parameters
% for the model.
%
% inputSize: the size of an input vector x^(i)
% numClasses: the number of classes 
% lambda: weight decay parameter
% inputData: an N by M matrix containing the input data, such that
%            inputData(:, c) is the cth input
% labels: M by 1 matrix containing the class labels for the
%            corresponding inputs. labels(c) is the class label for
%            the cth input
% options (optional): options
%   options.maxIter: number of iterations to train for

if ~exist('options', 'var')
    options = struct;
end

if ~isfield(options, 'maxIter')
    options.maxIter = 400;
end

if ~isfield(options, 'batchSize')
    options.batchSize = 100;
end

if ~isfield(options, 'learningRate')
    options.learningRate = 0.1;
end

numCases = size(inputData, 2);
batchSize = options.batchSize;
numbatches = numCases / batchSize;

% initialize parameters
theta = 0.005 * randn(numClasses * inputSize, 1);

for i=1:options.maxIter
    perm = randperm(numCases);
    trainData = inputData(:, perm);
    trainLabels = labels(perm);
    for l = 1 : numbatches
        batchData = trainData(:, (l - 1) * batchSize + 1 : l * batchSize);
        batchLabels = trainLabels((l - 1) * batchSize + 1 : l * batchSize);
        [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, batchData, batchLabels);
        theta = theta - options.learningRate * grad;
    end
    disp( ['iter:' num2str(i) ' cost:' num2str(cost)] );
end
softmaxOptTheta = theta;

% Fold softmaxOptTheta into a nicer format
softmaxModel.optTheta = reshape(softmaxOptTheta, numClasses, inputSize);
softmaxModel.inputSize = inputSize;
softmaxModel.numClasses = numClasses;
                          
end                          
