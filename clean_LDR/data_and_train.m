%% Load and Prepare Training Data
load("combined_1000.mat")

X = X(1:2000, :);

X = (X - mean(X, 2)) ./ std(X, 0, 2);  % Z-score normalization (axis 1)

% --- Split Data into Training and Validation Sets ---
% Split 80% of data for training and 20% for validation
% Shuffle indices
numData = size(X, 1);

% Define split ratios
valRatio = 0.25;

cv = cvpartition(numData, "HoldOut", valRatio);

% Split data
X_train = X(cv.training, :);
X_val = X(cv.test, :);


%% --- Define Encoder-Decoder Network (Autoencoder) ---
inputSize = 1000;  % Sequence length (number of features per time step)

% Training
latentDims = [64, 32, 16, 8];
kernel_sizes = [5 3; 3 3]; 
filter_sizes = [32 16 8 4; 16 8 4 2];
power_c = [1, 0.75, 0.5, 0.25, 0];

max_power_of_signals = max(mean(X_train.^2, 2));

sigma = sqrt(power_c*max_power_of_signals);

w = zeros(2, 2, size(kernel_sizes, 2)*size(filter_sizes, 2));
l = zeros(8, 1);

for i=1:size(w, 3)
    kernel_idx = floor(i/5)+1; filter_idx = mod(i-1, 4)+1; 
    latent_idx = filter_idx;
    
    w(:, :, i) = [kernel_sizes(:, kernel_idx) filter_sizes(:, filter_idx)];
    l(i) = latentDims(latent_idx);
end

networks = cell(8*5, 1);

%% Load models and initialize
num_models = numel(networks);
num_sigs = length(sigma);
CORs = zeros(num_sigs, 1);

X_train_augs = cell(num_sigs, 1);

% Compute correlation scores between original and augmented data
for n = 1:num_sigs
    X_train_aug = augmentData(X_train, sigma(n));
    X_train_augs{n} = X_train_aug;
    for j = 1:size(X_train, 1)
        CORs(n) = CORs(n) + corr(X_train(j, :)', X_train_aug(j, :)') / size(X_train, 1);
    end
end


save("train_data.mat", "X_train_augs", "X_train", "X_val");


% Base folder for saving all model subfolders
base_folder = 'reconstruction_quality_figures';
if ~exist(base_folder, 'dir')
    mkdir(base_folder);
end


% Loop over each trained model
for i = 1:num_models
    model_set = networks{i};
    net = model_set{1};           % dlnetwork
    training_fig = model_set{3};  % Training info (assumed to be a figure handle)

    % Create a subfolder for each model
    model_folder = fullfile(base_folder, sprintf('model_%02d', i));
    if ~exist(model_folder, 'dir')
        mkdir(model_folder);
    end

    mean_MSE_per_bin = zeros(num_sigs, 1);
    std_MSE_per_bin = zeros(num_sigs, 1);

    for s = 1:num_sigs
        X_aug = augmentData(X_train, sigma(s));
        recon = predict(net, X_aug);
        recon_errors = mean((recon - X_aug).^2, 2);

        mean_MSE_per_bin(s) = mean(recon_errors);
        std_MSE_per_bin(s) = std(recon_errors);
    end

    % Plotting reconstruction quality
    fig = figure('Visible', 'off');
    CORs_adjusted = 1 - CORs;
    CORs_adjusted(abs(CORs_adjusted) < 1e-2) = 0;

    x_data = CORs_adjusted * 100;
    y_data = mean_MSE_per_bin;
    
    x_marginu = 0.1 * max(x_data);
    y_marginu = 0.1 * max(y_data);
    x_marginl = 0.1 * min(x_data);
    y_marginl = 0.1 * min(y_data);

    errorbar(x_data, y_data, std_MSE_per_bin, 'o-', 'LineWidth', 2);
    xlim([min(x_data) - x_marginl - range(x_data)/10, max(x_data) + x_marginu]);
    ylim([min(y_data) - y_marginl - range(y_data)/10, max(y_data) + y_marginu]);
    xlabel('Eğitim Verisinden Sapma (%)');
    ylabel('Ortalama Yeniden İnşa MSE');
    title(sprintf('Yeniden İnşa Kalitesi vs. Girdi Sapması - Model %d', i));
    grid on;

    % Save reconstruction quality figure
    saveas(fig, fullfile(model_folder, sprintf('Model_%02d_ReconQuality_vs_Deviation.fig', i)));
    close(fig);

    %show(training_fig);                     % Display training progress
end


%%
load train_data.mat

for n=1:8*5
    sigma_idx = mod(n-1, 5)+1;
    other_idx = mod(n-1, 8)+1;

    % Augment training data
    X_train_aug = X_train_augs{sigma_idx};
    
    % Convert to cell arrays of sequences
    X_train_aug_cell = num2cell(X_train_aug, 2);  % 1 cell per sequence
    X_train_cell = num2cell(X_train, 2);
    X_val_cell = num2cell(X_val, 2);



    latentDim = l(other_idx);
    batch_size = 32;
    
    autoencoderLayers = [
        sequenceInputLayer(inputSize,"Name","input")
    
        % Encoder
        convolution1dLayer(w(1, 1, other_idx),w(1,2, other_idx),"Name","conv1","Padding","same")
        batchNormalizationLayer("Name","bn1")
        reluLayer("Name","relu1")
        dropoutLayer(0.1,"Name","dropout1")
    
        convolution1dLayer(w(2,1,other_idx),w(2,2, other_idx),"Name","conv2","Padding","same")
        batchNormalizationLayer("Name","bn2")
        reluLayer("Name","relu2")
        dropoutLayer(0.1,"Name","dropout2")
        
        fullyConnectedLayer(latentDim,"Name","bottleneck") % Bottleneck latentdim
    
        % Decoder
        transposedConv1dLayer(w(2,1, other_idx),w(2,2, other_idx),"Name","tconv3","Cropping","same")
        batchNormalizationLayer("Name","bn7")
        reluLayer("Name","relu7")
        dropoutLayer(0.1,"Name","dropout6")
    
        transposedConv1dLayer(w(1,1, other_idx),w(1,2, other_idx),"Name","tconv4","Cropping","same")
        batchNormalizationLayer("Name","bn8")
        reluLayer("Name","relu8")
        dropoutLayer(0.1,"Name","dropout7")
        
        fullyConnectedLayer(inputSize, "Name", "output_fc")  % Match original input size
    ];
    
    metric = rmseMetric(NormalizationFactor="batch-size");
    validation_freq = floor(size(X_train_aug_cell, 1)/batch_size);
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 100000, ...
        'MiniBatchSize', batch_size, ...
        'InitialLearnRate', 1e-4, ...
        'LearnRateDropFactor', 0.1, ...
        'GradientThresholdMethod','absolute-value', ...
        'GradientThreshold', 1, ...
        'Metrics', metric, ...
        'Plots', 'none', ...
        'L2Regularization', 1e-5, ...
        'OutputNetwork', 'best-validation', ...
        'Shuffle', 'every-epoch', ...
        'ValidationPatience', 100, ...
        'Verbose', false, ...
        'VerboseFrequency', validation_freq, ...
        'ValidationData', {X_val_cell, X_val_cell}, ...
        'ValidationFrequency', validation_freq);
    
    % --- Train the Autoencoder Network ---
    start_time = tic;
    [autoencoder, info] = trainnet(X_train_aug_cell, X_train_cell, autoencoderLayers, "mse", options);
    
    end_time = toc(start_time);
    fprintf('Finished training network (%d) in %.4f seconds\n', n, end_time);

    % Define a new dlnetwork that stops at the bottleneck
    % Identify the name of the bottleneck layer: 'bottleneck'
    % Create a layerGraph from the trained autoencoder
    fullLG = layerGraph(autoencoder);
    
    % Cut decoder layers
    encoderLG = removeLayers(fullLG, {'tconv3', 'bn7', 'relu7', 'dropout6', 'tconv4', 'bn8', 'relu8', 'dropout7', 'output_fc'});
    
    % Add a dummy regression output to make the encoder valid
    encoderLG = addLayers(encoderLG, regressionLayer("Name","output"));
    encoderLG = connectLayers(encoderLG, "bottleneck", "output");
    
    % Assemble network
    encoderNet = assembleNetwork(encoderLG);
    
    % --- Evaluate the Model ---
    X_cell = X;
    
    % Forward pass on validation data to get reconstruction
    predicted = predict(autoencoder, X_cell);
    
    recon_errors = zeros(1, size(X_cell, 1));
    
    for i = 1:size(X_cell, 1)
        recon_errors(i) = mse(predicted(i, :), X_cell(i, :));
    end
    
    % Choose threshold — e.g., 97th percentile (adjust this!)
    threshold = prctile(recon_errors, 97);  % or try 95, 98, etc.
    
    %fprintf('Chosen rejection threshold (97th percentile): %.6f\n', threshold);
    
    % Perform clustering (Enhanced Hybrid + Label Matching)
    
    % Forward pass for entire dataset
    embeddings = [];  % Will hold NxD projection outputs
    batchSizeInfer = 128;
    numBatchesInfer = ceil(size(X, 1) / batchSizeInfer);
    
    for i = 1:numBatchesInfer
        idx = (i-1)*batchSizeInfer + 1 : min(i*batchSizeInfer, size(X, 1));
        X_batch = X(idx, :);
        emb = predict(encoderNet, X_batch');  % Use the full network for the forward pass
        embeddings = [embeddings; emb'];  % Store N x D
    end
    
    % Use KMeans clustering (k=3) on latent embeddings
    k = 2;
    [labels, centroids_cell] = kmeans(embeddings, k, 'Distance', 'cosine', 'Replicates', 5);
    
    
    % Infer true class labels by segment index
    true_class = zeros(size(labels));
    true_class(1:1000) = 1;
    true_class(1001:2000) = 2;
    
    % Map kmeans labels to true classes using majority vote
    mapping = zeros(1, k);
    for class_id = 1:k
        idx_range = (class_id-1)*1000 + 1 : class_id*1000;
        majority_label = mode(labels(idx_range));
        mapping(majority_label) = class_id;
    end
    
    % In case of label collisions (e.g., same majority label mapped to two classes)
    % Use Hungarian algorithm for optimal assignment
    if length(unique(mapping)) < k
        disp("Collision in label assignment. Resolving with Hungarian algorithm.");
        costMatrix = zeros(k);
        for i = 1:k
            for j = 1:k
                range_i = (i-1)*1000 + 1 : i*1000;
                costMatrix(i, j) = -sum(labels(range_i) == j);  % negative count = cost
            end
        end
        [assignments, ~] = munkres(costMatrix);
        for j = 1:k
            mapping(j) = assignments(j);
        end
    end
    
    % Remap predicted labels to class names (1=Engin, 2=Idris, 3=Taha)
    mapped_labels = zeros(size(labels));
    for i = 1:k
        mapped_labels(labels == i) = mapping(i);
    end
   

    networks{n} = {autoencoder, encoderNet, info, end_time, centroids_cell, mapped_labels, threshold};
end

%%
for n=1:40
    network = networks{n};
    ValidationHistory = network{3}.ValidationHistory.Loss;
    Iteration = network{3}.ValidationHistory.Iteration;
    StopIteration = network{3}.OutputNetworkIteration;
    %training_history = 
    network{end+1} = ValidationHistory(Iteration == StopIteration);
    
    networks{n} = network;
end

% Save the cluster centroids and distance metric
% Save the kmeans model
save("clustering_models_SNR_inc.mat", "networks");

%%
function X_aug = augmentData(X, sigma)    
    X_aug = X + sigma * randn(size(X));
end


