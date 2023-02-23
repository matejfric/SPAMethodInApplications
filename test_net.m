clear all

%net = alexnet;
net = resnet18;
sz = net.Layers(1).InputSize; % size of the first layer

%analyzeNetwork(net)

I = imread('peppers.png');
I = imresize(I,sz(1:2));

% get descriptors
layer_name = 'pool5';
featuresTrain = activations(net,I,layer_name,'OutputAs','rows');