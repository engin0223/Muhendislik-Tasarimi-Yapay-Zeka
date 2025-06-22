A = load("combined_1000.mat").X;

numVectors = 1000;
vectorSize = 1000;

X = A;

X = (X - mean(X, 2)) ./ std(X, 0, 2);  % Z-score normalization (axis 1)

%% Data Creation
max_power_of_signals = max(mean(X(1:2000, :).^2, 2));

lowest_SNR = -5; highest_SNR = 30;

lower_bound_for_sigma = sqrt(10^-(highest_SNR/10)*max_power_of_signals);
upper_bound_for_sigma = sqrt(10^-(lowest_SNR/10)*max_power_of_signals);

mu = 0;
s = linspace(lower_bound_for_sigma, upper_bound_for_sigma, 60);

noise_number = length(mu)*length(s);
signal_number = size(X, 1);

SNR_number = noise_number*signal_number;

SNRs = zeros(noise_number, signal_number);

y = generateGaussians(mu, s, 1000);

noisy_signals = zeros(noise_number, signal_number, 1000);

tracker = ProgressTracker(signal_number*noise_number);

dp = tracker.getQueue();

tracker.start();

batch_size = 1000;


for k=0:floor(signal_number/batch_size)
    start = k*(batch_size)+1; finish=min((k+1)*batch_size, signal_number);
    parfor j=start:finish
        for i=1:noise_number
            noise = y(:, i)'; signal = X(j, :);
            SNRs(i, j) = compute_snr(signal, signal+noise, noise);
            CORs(i, j) = corr(signal(:), signal(:)+noise(:));
            noisy_signals(i, j, :) = signal+noise;
    
            send(dp, 1)
        end
    end
end


% Assuming SNRs is already computed
figure;
imagesc(SNRs);              % Create the heatmap
colormap("hot");            % Choose a colormap (you can use 'parula', 'jet', etc.)
colorbar;                   % Show colorbar
xlabel('Signal Index');
ylabel('Noise Index');
title('SNR Heatmap');

% Assume A is of size 60x3000x1000
[rows, cols, pages] = size(noisy_signals);

% Preallocate output
X_test = zeros(rows * cols, pages);

for i = 1:cols
    % Extract page i: size 60x3000
    temp = squeeze(noisy_signals(:,i,:));
    
    % Reshape by traversing columns first (i.e., dim 2, then dim 1)
    X_test((i-1)*60+1:i*60, :) = temp;
end

noise_sigmas = s;

save("test_data.mat", "SNRs", "CORs", "X_test", "noise_sigmas")

%%
load("clustering_models_SNR_inc.mat")
load("test_data.mat")


%%

loss_vector = zeros(40,1);      % To store validation losses
param_vector = zeros(40,1);     % To store learnable parameter counts

current_info = networks{1}{3};
current_net = networks{1}{1};   % Assuming the actual network is here
current_outnet = current_info.OutputNetworkIteration;
current_valhistory = current_info.ValidationHistory;

% Get number of learnable parameters
params = current_net.Learnables.Value;
total_params = sum(cellfun(@numel, params));
param_vector(1) = total_params;

% Get initial loss and index
best_loss = current_valhistory.Loss(current_valhistory.Iteration == current_outnet);
best_netidx = 1;
loss_vector(1) = best_loss;

for n = 2:40
    current_info = networks{n}{3};
    current_net = networks{n}{1};
    current_outnet = current_info.OutputNetworkIteration;
    current_valhistory = current_info.ValidationHistory;
    current_loss = current_valhistory.Loss(current_valhistory.Iteration == current_outnet);
    
    % Store loss
    loss_vector(n) = current_loss;

    % Count learnable parameters
    params = current_net.Learnables.Value;
    total_params = sum(cellfun(@numel, params));
    param_vector(n) = total_params;

    % Update best if necessary
    if current_loss < best_loss
        best_loss = current_loss;
        best_netidx = n;
    end
end


%%

norm_loss = (loss_vector - min(loss_vector)) / (max(loss_vector) - min(loss_vector));
norm_param = (param_vector - min(param_vector)) / (max(param_vector) - min(param_vector));

combined_score = norm_loss + norm_param;  % Equal weight to loss and size

[~, best_idx] = min(combined_score);

fprintf('Best network index: %d\n', best_idx);
fprintf('Validation Loss: %.4f\n', loss_vector(best_idx));
fprintf('Parameter Count: %d\n', param_vector(best_idx));

%%
base_folder = 'performance_results';
X_test = gpuArray(X_test);

best_n = 1;

best_f1 = 0;
best_idx = 1;

best_f1_p = 0;
best_idx_p = 1;

best_f1_pp = 0;
best_idx_pp = 1;

best_confMat = 0;


for n = 37
    % Bu iterasyon için alt klasör oluştur
    iter_folder = fullfile(base_folder, sprintf('Network n%d', n));
    if ~exist(iter_folder, 'dir')
        mkdir(iter_folder);
    end
    
    autoencoder = networks{n}{1};
    start_time = tic;
    
    % Tahmin yap ve yeniden inşa hatalarını hesapla
    predicted_new = predict(autoencoder, X_test);
    fprintf("Tahminler %.4f saniyede tamamlandı\n", toc(start_time))
    
    start_time2 = tic;

    % Compute squared differences
    squared_diffs = (predicted_new - X_test).^2;
    
    % Compute mean squared error per row (sample)
    recon_errors_noisy = mean(squared_diffs, 2);
    
    % Move from GPU to CPU if needed for saving/plotting
    recon_errors_noisy = gather(recon_errors_noisy);
    
    % Transpose to match original 1D row vector format
    recon_errors_noisy = recon_errors_noisy';

    fprintf("Hata hesaplama %.4f saniyede tamamlandı\n", toc(start_time2))

    % Boyutlar
    [noise_number, signal_number] = size(SNRs);
    
    recon_errors_remapped = reshape(recon_errors_noisy, noise_number, signal_number);

    best_f1_p = 0;

    encoder = networks{n}{2};
    
    embeddings = predict(encoder, X_test');

    embeddings = reshape(gather(embeddings), [], noise_number, signal_number);

    centroids_cell = networks{n}{5};

    start_time3 = tic;

    for nidx=1:60
        recon_error_for_SNR = recon_errors_remapped(nidx, :);

        % Take the first 2000 values (or fewer if recon_error_for_SNR has less than 2000)
        subset_for_threshold = recon_error_for_SNR(1:min(2000, length(recon_error_for_SNR)));
    
        embeddings2 = squeeze(embeddings(:, nidx, :));
    
        num_samples = size(embeddings2, 2);
        num_centroids = size(centroids_cell, 1);
        
        predicted_labels_cosine = zeros(num_samples, 1);
        
        for j=90:100
            % Calculate 90th percentile threshold
            threshold = prctile(subset_for_threshold, j);
    
            for i = 1:num_samples
                recon_err = recon_error_for_SNR(i);
                if recon_err < threshold
                    sample_vec = embeddings2(:, i);  % 1x64
                    
                    % Compute cosine distances to each centroid
                    distances = zeros(num_centroids, 1);
                    for c = 1:num_centroids
                        centroid_vec = centroids_cell(c, :);
                        
                        % cosine similarity = dot(a,b)/(||a||*||b||)
                        cos_sim = dot(sample_vec, centroid_vec) / (norm(sample_vec) * norm(centroid_vec));
                        
                        % cosine distance = 1 - cosine similarity
                        distances(c) = 1 - cos_sim;
                    end
                    
                    % Assign label to closest centroid (lowest cosine distance)
                    [~, min_idx] = min(distances);
                    predicted_labels_cosine(i) = min_idx;
                else
                    predicted_labels_cosine(i) = 3;
                end
            end
        
            % Infer true class labels by segment index
            true_class = zeros(3000, 1);
            true_class(1:1000) = 1;
            true_class(1001:2000) = 2;
            true_class(2001:3000) = 3;
            
            confMat = confusionmat(true_class, predicted_labels_cosine);
            
            
            % For class 1 (index 1)
            TP_class1 = confMat(1,1);
            FP_class1 = sum(confMat(:,1)) - TP_class1;
            FN_class1 = sum(confMat(1,:)) - TP_class1;
            
            % For class 2 (index 2)
            TP_class2 = confMat(2,2);
            FP_class2 = sum(confMat(:,2)) - TP_class2;
            FN_class2 = sum(confMat(2,:)) - TP_class2;
            
            % For class 3 (index 3)
            TP_class3 = confMat(3,3);
            FP_class3 = sum(confMat(:,3)) - TP_class3;
            FN_class3 = sum(confMat(3,:)) - TP_class3;
            
            % Precision and Recall for each class
            precision_class1 = TP_class1 / (TP_class1 + FP_class1 + eps);
            recall_class1    = TP_class1 / (TP_class1 + FN_class1 + eps);
            
            precision_class2 = TP_class2 / (TP_class2 + FP_class2 + eps);
            recall_class2    = TP_class2 / (TP_class2 + FN_class2 + eps);
            
            precision_class3 = TP_class3 / (TP_class3 + FP_class3 + eps);
            recall_class3    = TP_class3 / (TP_class3 + FN_class3 + eps);
            
            % F1 score for each class
            f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1 + eps);
            f1_class2 = 2 * (precision_class2 * recall_class2) / (precision_class2 + recall_class2 + eps);
            f1_class3 = 2 * (precision_class3 * recall_class3) / (precision_class3 + recall_class3 + eps);
            
        
            % Macro F1 Score
            macro_f1_score = (f1_class1 + f1_class2 + f1_class3) / 3;
        
            if macro_f1_score > best_f1
                best_f1 = macro_f1_score;
                best_idx = j;
                best_confMat = confMat;
            end
        end
        
        best_f1_p = best_f1_p + best_f1*(61-nidx);
    end
    

    fprintf("En iyi Kümeleme Hesaplama %.4f saniyede tamamlandı\n", toc(start_time3))


    disp([best_idx, best_f1_p])
    disp(best_confMat)
    
    if best_f1_p > best_f1_pp
        best_f1_pp = best_f1_p;
        best_n = n;
    end
    
    
end

%%
for n = 1:40
    % Bu iterasyon için alt klasör oluştur
    iter_folder = fullfile(base_folder, sprintf('Network n%d', n));
    if ~exist(iter_folder, 'dir')
        mkdir(iter_folder);
    end
    
    autoencoder = networks{n}{1};
    start_time = tic;
    
    % Tahmin yap ve yeniden inşa hatalarını hesapla
    predicted_new = predict(autoencoder, X_test);
    fprintf("Tahminler %.4f saniyede tamamlandı\n", toc(start_time))
    
    start_time2 = tic;

    % Compute squared differences
    squared_diffs = (predicted_new - X_test).^2;
    
    % Compute mean squared error per row (sample)
    recon_errors_noisy = mean(squared_diffs, 2);
    
    % Move from GPU to CPU if needed for saving/plotting
    recon_errors_noisy = gather(recon_errors_noisy);
    
    % Transpose to match original 1D row vector format
    recon_errors_noisy = recon_errors_noisy';

    fprintf("Hata hesaplama %.4f saniyede tamamlandı\n", toc(start_time2))

    % Boyutlar
    [noise_number, signal_number] = size(SNRs);
    
    recon_errors_remapped = reshape(recon_errors_noisy, noise_number, signal_number);

    % --- After you compute CORs and recon_errors_remapped (size: noise_number x signal_number) ---

    % 1. Rescale Pearson's r to deviation %
    % Assuming CORs are mostly positive (0 to 1), if negatives exist, clip them or handle separately.
    CORs_adjusted = 1 - CORs;
    CORs_adjusted(abs(CORs_adjusted) < 1e-2) = 0;

    deviation_percent = 100 * CORs_adjusted;  % 0% = r=1, 100% = r=0
    
    % 2. Define deviation bins, e.g. 10 bins of 10% each from 0 to 100%
    bin_edges = 0:10:100;  % 0-10%, 10-20%, ..., 90-100%
    num_bins = length(bin_edges) - 1;
    
    % Preallocate arrays to store mean and std MSE per bin
    mean_MSE_per_bin = zeros(num_bins, 1);
    std_MSE_per_bin = zeros(num_bins, 1);
    bin_centers = (bin_edges(1:end-1) + bin_edges(2:end)) / 2;
    
    % Flatten data for easier indexing
    deviation_vec = deviation_percent(:);
    mse_vec = recon_errors_remapped(:);
    
    for b = 1:num_bins
        % Find indices of data points in the current bin
        in_bin = deviation_vec >= bin_edges(b) & deviation_vec < bin_edges(b+1);
        
        % Calculate statistics if bin not empty
        if any(in_bin)
            mean_MSE_per_bin(b) = mean(mse_vec(in_bin));
            std_MSE_per_bin(b) = std(mse_vec(in_bin));
        else
            mean_MSE_per_bin(b) = NaN;
            std_MSE_per_bin(b) = NaN;
        end
    end
    
    % 3. Plot Mean MSE vs. Deviation with error bars
    fig_i = figure('Visible', 'off');
    errorbar(bin_centers, mean_MSE_per_bin, std_MSE_per_bin, 'o-', 'LineWidth', 2);
    xlabel('Eğitim Verisinden Sapma (%)');
    ylabel('Ortalama Yeniden İnşa MSE');
    title('Yeniden Oluşturma Kalitesi ve Girdi Sapması İlişkisi');
    grid on;
    
    
    % Optional: Save the plot
    saveas(fig_i, fullfile(iter_folder, 'MSE_vs_Deviation.fig'));

    start_time3 = tic;
    
    % İndeks ızgarası oluştur
    [SignalIdx, NoiseIdx] = meshgrid(1:signal_number, 1:noise_number);
    
    % Scatter için veriyi düzleştir
    x = SignalIdx(:);
    y = SNRs(:);
    z = recon_errors_remapped(:);  % Z ekseni: yeniden inşa hatası
    colorData = CORs(:);           % Renk verisi: Korelasyon katsayısı

    % --- Eşik yüzeyli 3B saçılım grafiği ---
    fig1 = figure('Visible', 'off');
    hold on;
    scatter3(x, y, z, 36, colorData, 'filled');  % z yerine log(z) de kullanılabilir
    colormap('jet');
    colorbar;
    xlabel('Sinyal İndeksi');
    ylabel('SNR Değeri');
    zlabel('Yeniden Oluşturma Hatası');
    title('3B Saçılım Grafiği: Yeniden Oluşturma Hatası');
    grid on;
    view(45, 30);
    hold off;
    set(gcf,'Visible','off','CreateFcn','set(gcf,''Visible'',''on'')');
    saveas(fig1, fullfile(iter_folder, '3Dscatter.fig'));
    close(fig1);

    % Scatter verisini kaydet
    save(fullfile(iter_folder, 'scatter_data.mat'), 'x', 'y', 'z', 'colorData');
    
    
    target_SNR = 30; % your target SNR value
    
    recon_error_for_SNR = recon_errors_remapped(target_SNR, :);

    % Take the first 2000 values (or fewer if recon_error_for_SNR has less than 2000)
    subset_for_threshold = recon_error_for_SNR(1:min(2000, length(recon_error_for_SNR)));

    snr_mask = (abs(SNRs-target_SNR) < 0.1);

    cor_for_SNR = CORs(snr_mask);

    encoder = networks{n}{2};

    embeddings = predict(encoder, X_test');

    embeddings = reshape(gather(embeddings), [], noise_number, signal_number);
    embeddings = squeeze(embeddings(:, target_SNR, :));
    centroids_cell = networks{n}{5};


    num_samples = size(embeddings, 2);
    num_centroids = size(centroids_cell, 1);
    
    predicted_labels_cosine = zeros(num_samples, 1);
    
    for j=90:100
        % Calculate 90th percentile threshold
        threshold = prctile(subset_for_threshold, j);

        for i = 1:num_samples
            recon_err = recon_error_for_SNR(i);
            if recon_err < threshold
                sample_vec = embeddings(:, i);  % 1x64
                
                % Compute cosine distances to each centroid
                distances = zeros(num_centroids, 1);
                for c = 1:num_centroids
                    centroid_vec = centroids_cell(c, :);
                    
                    % cosine similarity = dot(a,b)/(||a||*||b||)
                    cos_sim = dot(sample_vec, centroid_vec) / (norm(sample_vec) * norm(centroid_vec));
                    
                    % cosine distance = 1 - cosine similarity
                    distances(c) = 1 - cos_sim;
                end
                
                % Assign label to closest centroid (lowest cosine distance)
                [~, min_idx] = min(distances);
                predicted_labels_cosine(i) = min_idx;
            else
                predicted_labels_cosine(i) = 3;
            end
        end
    
        % Infer true class labels by segment index
        true_class = zeros(3000, 1);
        true_class(1:1000) = 1;
        true_class(1001:2000) = 2;
        true_class(2001:3000) = 3;
        
        confMat = confusionmat(true_class, predicted_labels_cosine);
        
        % Display confusion matrix
        disp('Confusion Matrix:');
        disp(confMat)
        
        % For class 1 (index 1)
        TP_class1 = confMat(1,1);
        FP_class1 = sum(confMat(:,1)) - TP_class1;
        FN_class1 = sum(confMat(1,:)) - TP_class1;
        
        % For class 2 (index 2)
        TP_class2 = confMat(2,2);
        FP_class2 = sum(confMat(:,2)) - TP_class2;
        FN_class2 = sum(confMat(2,:)) - TP_class2;
        
        % For class 3 (index 3)
        TP_class3 = confMat(3,3);
        FP_class3 = sum(confMat(:,3)) - TP_class3;
        FN_class3 = sum(confMat(3,:)) - TP_class3;
        
        % Precision and Recall for each class
        precision_class1 = TP_class1 / (TP_class1 + FP_class1 + eps);
        recall_class1    = TP_class1 / (TP_class1 + FN_class1 + eps);
        
        precision_class2 = TP_class2 / (TP_class2 + FP_class2 + eps);
        recall_class2    = TP_class2 / (TP_class2 + FN_class2 + eps);
        
        precision_class3 = TP_class3 / (TP_class3 + FP_class3 + eps);
        recall_class3    = TP_class3 / (TP_class3 + FN_class3 + eps);
        
        % F1 score for each class
        f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1 + eps);
        f1_class2 = 2 * (precision_class2 * recall_class2) / (precision_class2 + recall_class2 + eps);
        f1_class3 = 2 * (precision_class3 * recall_class3) / (precision_class3 + recall_class3 + eps);
        
    
        % Macro F1 Score
        macro_f1_score = (f1_class1 + f1_class2 + f1_class3) / 3;
    
        if macro_f1_score > best_f1
            best_f1 = macro_f1_score;
            best_idx = j;
        end
    end
    
    
    % Optional: plot confusion matrix (requires MATLAB R2017b or newer)
    fig4 = figure('visible', 'off');
    cm = confusionchart(confMat, ["Engin" "İdris" "Taha"]);
    sortClasses(cm, ["Engin" "İdris" "Taha"]);
    title('Belirsizlik Matrisi (Kosinüs Uzaklık Kümesi Atama)');
    saveas(fig4, fullfile(iter_folder, 'confmat.fig'));

    % --- Yeniden inşa hatası grafikleri (3 kişi için 3 parça) ---
    start_index = 1;
    split_sizes = 35*1000;
    labels = ["engin", "idris", "taha"];
    for i = 1:3
        end_index = start_index + split_sizes - 1;
        current_part = recon_errors_noisy(start_index:end_index);

        fig_part = figure('Visible', 'off');
        plot(current_part);
        title(['Bölüm ' num2str(i)]);
        set(gcf,'Visible','off','CreateFcn','set(gcf,''Visible'',''on'')');
        saveas(fig_part, fullfile(iter_folder, sprintf('recon_part_%d.fig', i)));
        close(fig_part);

        % Bölüm verisini kaydet
        save(fullfile(iter_folder, sprintf('recon_part_%d.mat', i)), ...
             'current_part');

        start_index = end_index + 1;
    end
    
    fprintf("Bu ağ için grafikler %.4f saniyede tamamlandı\n", toc(start_time3))
    fprintf("Bu ağ için her şey %.4f saniyede tamamlandı\n", toc(start_time))
    
    % --- Karışıklık matrisi grafiği ---
    %fig_conf = figure('Visible', 'off');
    %confusionchart(confMat, classNames);
    %title('Karışıklık Matrisi');
    %saveas(fig_conf, fullfile(iter_folder, 'confusion_matrix.png'));
    %close(fig_conf);

    % Karışıklık matrisi verisini kaydet
    %save(fullfile(iter_folder, 'confusion_matrix_data.mat'), 'confMat', 'classNames');
end




%%
base_folder = 'performance_results';

fig_files = dir(fullfile(base_folder, '**', '*.fig'));

fprintf("Found %d .fig files to update.\n", numel(fig_files));

for k = 1:numel(fig_files)
    if mod(k, 6) == 3
        fig_path = fullfile(fig_files(k).folder, fig_files(k).name);
        
        % Load figure invisibly (does not show figure)
        %fig = hgload(fig_path, 'Visible', 'off');
        fig = openfig(fig_path, "new", "invisible");

        fig.Children.XLabel = "Tahmin Edilen Sınıf";
        fig.Children.YLabel = "Doğru Sınıf";
        
        % Now set visibility on internally (without showing it)
        set(fig, 'Visible', 'on');
        
        % Save back to same file with visible on
        savefig(fig, fig_path);
        
        close(fig);
        fprintf("Updated: %s\n", fig_path);
    end
end


fprintf("All visible states updated.\n");


%%
function [assignment, cost] = munkres(costMat)
    [assignment, cost] = assignDeterministic(costMat);
end


function [assignment, cost] = assignDeterministic(costMat)
    n = size(costMat, 1);
    permsMat = perms(1:n);
    costs = zeros(size(permsMat, 1), 1);
    for i = 1:size(permsMat, 1)
        idx = sub2ind(size(costMat), 1:n, permsMat(i,:));
        costs(i) = sum(costMat(idx));
    end
    [cost, idx] = min(costs);
    assignment = permsMat(idx, :);
end


function gaussians = generateGaussians(mu_vec, sigma_vec, N)
    %GENERATEGAUSSIANS Generate Gaussian vectors for all (mu, sigma) combinations.
    %   gaussians = generateGaussians(mu_vec, sigma_vec, N)
    %   returns a matrix of size [num_mu * num_sigma, N] where each row
    %   contains samples from N(mu, sigma^2).

    if ~isvector(mu_vec) || ~isvector(sigma_vec)
        error('mu_vec and sigma_vec must be vectors.');
    end

    mu_vec = mu_vec(:);    % Ensure column vectors
    sigma_vec = sigma_vec(:);

    num_mu = length(mu_vec);
    num_sigma = length(sigma_vec);
    total_combinations = num_mu * num_sigma;

    gaussians = zeros(N, total_combinations);
    idx = 1;

    for i = 1:num_mu
        for j = 1:num_sigma
            mu = mu_vec(i);
            sigma = sigma_vec(j);
            gaussians(:, idx) = mu + sigma * randn(1, N); % Use randn for Gaussian
            idx = idx + 1;
        end
    end
end


function snr_db = compute_snr(original_signal, noisy_signal, noise)
    % Ensure input vectors are the same length
    if length(original_signal) ~= length(noisy_signal)
        error('Original and noisy signals must be the same length.');
    end
    
    
    % Compute power of signal and noise
    signal_power = mean(original_signal.^2);
    noise_power = mean(noise.^2);
    
    % Compute SNR in dB
    snr_db = 10 * log10(signal_power / noise_power);
end


function resetCounter()
    persistent iterCount
    iterCount = 0;
end


function updateCounter(~, start_time, n_windows)
    persistent iterCount
    if isempty(iterCount)
        iterCount = 0;
    end
    iterCount = iterCount + 1;

    if mod(iterCount, floor(n_windows/100)) == 0
        elapsed_time = toc(start_time);
        progress = iterCount / n_windows;
        remaining_time = elapsed_time / progress * (1 - progress);
        fprintf('Progress: %.2f%% | Elapsed: %s | Remaining: %s\n', ...
            progress*100, formatTime(elapsed_time), formatTime(remaining_time));
    end
end


function str = formatTime(seconds)
    if seconds < 60
        str = sprintf('%.1fs', seconds);
    elseif seconds < 3600
        str = sprintf('%.1fm', seconds/60);
    else
        str = sprintf('%.1fh', seconds/3600);
    end
end

%%
% Eşik altındaki noktalar için mantıksal maske
    belowThresholdMask = (z < threshold);
    
    % Eşik altı veriyi filtrele
    x_below = x(belowThresholdMask);
    y_below = y(belowThresholdMask);
    z_below = z(belowThresholdMask);
    color_below = colorData(belowThresholdMask);

    % --- 3B saçılım: eşik altı noktalar ---
    fig2 = figure('Visible', 'off');
    scatter3(x_below, y_below, log10(z_below), 36, color_below, 'filled');
    colormap('jet');
    colorbar;
    xlabel('Sinyal İndeksi');
    ylabel('Gürültü İndeksi');
    zlabel('log_{10}(Yeniden İnşa Hatası)');
    title('3B Saçılım: Eşik Altındaki Yeniden İnşa Hataları');
    grid on;
    view(45, 30);
    saveas(fig2, fullfile(iter_folder, '3Dscatter_below_threshold.png'));
    close(fig2);

    % Eşik altı veriyi kaydet
    save(fullfile(iter_folder, 'scatter_below_threshold.mat'), ...
         'x_below', 'y_below', 'z_below', 'color_below');


%%


