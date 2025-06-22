clear
load("network.mat", "autoencoder");  % loads the trained network
load("data_p.mat")

P = (P - mean(P, 2)) ./ std(P, 0, 2);  % Z-score normalization

%%
num_samples_P = size(P, 1);
inputSize = size(P, 2);  % should be 750

start_time = tic;
y_pred = predict(autoencoder, P)';

sprintf("Finished %d Samples in: %s", num_samples_P, formatTime(toc(start_time)))

%%
chunk_size = 10;
num_chunks = ceil(num_samples_P / chunk_size);
recon_error_pos = zeros(num_samples_P, 1, 'single');

% Create tracker instance
tracker = ProgressTracker(num_samples_P);

% Start the timer
tracker.start();
start_time = tic;

% Get the DataQueue
dp = tracker.getQueue();

for c = 1:num_chunks
    start_idx = (c-1)*chunk_size + 1;
    end_idx = min(c*chunk_size, num_samples_P);
    current_size = end_idx - start_idx + 1;
    
    x_chunk = P(start_idx:end_idx, :)';
    y_chunk = y_pred(:, start_idx:end_idx);
    
    xConst = parallel.pool.Constant(x_chunk);
    yConst = parallel.pool.Constant(y_chunk);
    
    % Preallocate chunk-local vector
    chunk_errors = zeros(current_size, 1, 'single');
    
    parfor i = 1:current_size
        chunk_errors(i) = mse(xConst.Value(:, i), yConst.Value(:, i));
        %sqrt(mean((xConst.Value(:, i) - yConst.Value(:, i)).^2))

        send(dp, i)
    end
    
    % After parfor, copy results to main output vector
    recon_error_pos(start_idx:end_idx) = chunk_errors;
    
    clear xConst yConst
end

%%
load("datasets.mat");


%%
recon_error_neg = [];

for i=1:4
    X = datasets{i};
    valid = valid_start{i};
    num_samples = length(valid);
    offset = length(recon_error_neg);
    
    chunk_size = 750 * 150 - 1;
    
    % Precompute contiguous regions where valid is true
    valid_diff = diff([0; valid(:); 0]); % pad to catch edges
    start_inds = find(valid_diff == 1);
    end_inds   = find(valid_diff == -1) - 1;
    
    % Create tracker instance
    tracker = ProgressTracker(num_samples);
    tracker.start();
    tracker.reset();
    start_time = tic;
    dp = tracker.getQueue();
    
    for r = 1:length(start_inds)
        region_start = start_inds(r);
        region_end = end_inds(r);
        region_len = region_end - region_start + 1;
    
        % Compute number of full chunks in this region
        num_chunks = floor(region_len / chunk_size);
    
        for c = 0:(num_chunks - 1)
            start_idx = region_start + c * chunk_size;
            end_idx = start_idx + chunk_size - 1;
            send(dp, end_idx-start_idx);  % Report progress
    
            x_chunk = gpuArray(buffer(X(start_idx:min(end_idx+749, end), 1), 750, 749, "nodelay"))';
            x_chunk = (x_chunk - mean(x_chunk, 2)) ./ std(x_chunk, 0, 2);
            y_chunk = predict(autoencoder, x_chunk);
    
            %disp(end_idx-start_idx)
    
            x_chunk = gather(x_chunk);
            y_chunk = gather(y_chunk);
    
            % Calculate squared differences
            diffs = x_chunk - y_chunk;             % [nSamples x current_size]
            squared_diffs = diffs .^ 2;            % Elementwise square
            
    
            % After parfor, copy results to main output vector
            recon_error_neg(offset+start_idx:offset+end_idx) = mean(squared_diffs, 2)'; 
        end
    
        % Optional: Process leftover chunk at the end of the region
        remainder = mod(region_len, chunk_size);
        if remainder > 0
            start_idx = region_end - remainder + 1;
            end_idx = region_end;
            send(dp, end_idx-start_idx);  % Report progress
    
            x_chunk = gpuArray(buffer(X(start_idx:min(end_idx+749, end), 1), 750, 749, "nodelay"))';
            x_chunk = (x_chunk - mean(x_chunk, 2)) ./ std(x_chunk, 0, 2);
            y_chunk = predict(autoencoder, x_chunk);
    
            %disp(end_idx-start_idx)
    
            x_chunk = gather(x_chunk);
            y_chunk = gather(y_chunk);
    
            % Calculate squared differences
            diffs = x_chunk - y_chunk;             % [nSamples x current_size]
            squared_diffs = diffs .^ 2;            % Elementwise square
            
    
            % After parfor, copy results to main output vector
            recon_error_neg(offset+start_idx:offset+end_idx) = mean(squared_diffs, 2)'; 
        end
        
    end
end


%% Test
num_neg = length(recon_error_neg);
recon_error = [recon_error_neg; recon_error_pos];


% Plot
figure;
histogram(recon_error_neg, 100, 'Normalization', 'probability', 'FaceColor', 'b', 'DisplayName', 'Negative');
hold on;
histogram(recon_error_pos, 100, 'Normalization', 'probability', 'FaceColor', 'r', 'DisplayName', 'Positive');
legend;
xlabel('Reconstruction Error');
ylabel('Probability');
title('Reconstruction Error Distribution');


% Ground truth labels: 0 for negatives, 1 for positives
labels = logical([zeros(num_neg,1); ones(156,1)]);

% Calculate threshold from 97th percentile of *positive* reconstruction errors
pos_errors = recon_error(labels == 1);
threshold = prctile(pos_errors, 90);

% Predicted labels
pred_labels = recon_error <= threshold;

% Confusion chart
figure;
cm = confusionchart(labels, pred_labels);
cm.Title = sprintf("Confusion Chart (Threshold = %.4f)", threshold);

% Optional: plot histogram with threshold
figure;
hold on;
histogram(recon_error(labels==0), 100, 'DisplayName', 'Negatives', 'FaceColor', [0.2 0.6 1]);
histogram(recon_error(labels==1), 100, 'DisplayName', 'Positives', 'FaceColor', [1 0.4 0.4]);
xline(threshold, '--k', 'LineWidth', 2, 'DisplayName', '97th percentile threshold');
legend;
xlabel('Reconstruction Error');
ylabel('Count');
title('Reconstruction Error Distribution');

% True labels and predictions already available
% labels: true labels (0 = negative, 1 = positive)
% pred_labels: predicted labels

% Confusion matrix values
TP = sum((labels == 1) & (pred_labels == 1));
TN = sum((labels == 0) & (pred_labels == 0));
FP = sum((labels == 0) & (pred_labels == 1));
FN = sum((labels == 1) & (pred_labels == 0));

% --- Metrics for positive class (label = 1)
precision_pos = TP / (TP + FP);
recall_pos = TP / (TP + FN);
f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos);

% --- Metrics for negative class (label = 0)
precision_neg = TN / (TN + FN);
recall_neg = TN / (TN + FP);
f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg);

% Display results
fprintf("Positive Class (1):\n");
fprintf("  Precision: %.4f\n", precision_pos);
fprintf("  Recall:    %.4f\n", recall_pos);
fprintf("  F1 Score:  %.4f\n", f1_pos);

fprintf("\nNegative Class (0):\n");
fprintf("  Precision: %.4f\n", precision_neg);
fprintf("  Recall:    %.4f\n", recall_neg);
fprintf("  F1 Score:  %.4f\n", f1_neg);
