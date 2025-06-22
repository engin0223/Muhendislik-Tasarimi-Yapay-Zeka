% Load your .mat file
load('train_data.mat');  % Replace with your actual variable names

%% Let's assume your input is in variable `X`
X = permute(single(), [1 3 2]);  % Shape as (N, 1, 1000)

% Save as .npy using MATLAB's Python interface
py.numpy.save('train_data_val_stm32.npy', py.numpy.array(X))


