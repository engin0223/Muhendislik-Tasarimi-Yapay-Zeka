%% Load and Prepare Training Data
load("combined_1000.mat")

X = X(1:2000, :);

X = (X - mean(X, 2)) ./ std(X, 0, 2);  % Z-score normalization (axis 1)

load("clustering_models_SNR_inc.mat")

%% Qualitative and Quantitative Check for All Models
base_folder = 'latent_quality_figures';
metrics = zeros(40, 3); % Column 1-2 Silhouette Score (class 1 and class 2), Column 2 Davies-Bouldin Index

% Set the random seed for reproducibility
rng('default'); 


for i=1:40
    % Create a subfolder for each model
    %model_folder = fullfile(base_folder, sprintf('model_%02d', i));
    %if ~exist(model_folder, 'dir')
        %mkdir(model_folder);
    %end

    network = networks{i}{2};
    
    L = predict(network, X');
    %YL = tsne(L', "NumDimensions", 2);
    
    %x = YL(:,1);
    %y = YL(:,2);
    %z = YL(:,3);
    group = [repmat("engin", [1 1000]) repmat("idris", [1 1000])]';

    % What are those unique groups? 
    %uniqueGroups = ["Engin" "İdris"]; 
    %colors = ["blue" "red"];

    % Initialize some axes
    %view(3);
    %grid on
    %hold on

    % Plot each group individually: 
    %for k = 1:length(uniqueGroups)
          % Get indices of this particular unique group:
          %ind = group==uniqueGroups(k); 
          
          % Plot only this group: 
          %plot3(x(ind),y(ind),z(ind),'.','color',colors(k),'markersize',20); 
    %end
    %legend('Engin','İdris')

    %fig = gcf;

    %saveas(fig, fullfile(model_folder, "Qualitative.fig"))
   
    %fig2 = figure;

    s = silhouette(L', group, 'cosine');

    s1 = mean(s(1:1000));
    s2 = mean(s(1001:2000));
    DB = daviesBouldin(L', [ones(1, 1000) 2*ones(1, 1000)]);

    metrics(i, :) = [s1 s2 DB];
end

%%

min_class_s_normalized = (min(metrics(:, 1:2)')+1)/2;
DBI_normalized = 1 - (metrics(:, 3) - min(metrics(:, 3))) / (max(metrics(:, 3))-min(metrics(:, 3)));

unified_score = 0.5 * min_class_s_normalized + 05 * DBI_normalized';

recon_err = zeros(1, 40);

for i=1:40
    recon_err(i) = networks{i}{end};
end

%%
modelIDs = sprintfc('Model %d', 1:40);
plotParetoFront(recon_err, unified_score, modelIDs);

bestModelID = selectBestParetoPoint(recon_err, unified_score, modelIDs);

fprintf('The deterministically selected best model is: %s\n', bestModelID);

function plotParetoFront(all_x, all_y, all_ids)
    % plotParetoFront: Creates a Pareto front plot for multi-objective optimization.
    %
    % Inputs:
    %   all_x   - A vector of values for the x-axis (e.g., Reconstruction Error).
    %             LOWER is considered BETTER.
    %   all_y   - A vector of values for the y-axis (e.g., Unified Cluster Score).
    %             HIGHER is considered BETTER.
    %   all_ids - A cell array of strings with labels for each point (optional).

    % --- Step 1: Identify the Pareto Optimal Points ---
    num_points = length(all_x);
    is_pareto = true(num_points, 1); % Assume all points are on the front initially

    for i = 1:num_points
        % Check if any other point dominates point 'i'
        for j = 1:num_points
            if i == j
                continue; % Don't compare a point to itself
            end
            
            % A point 'j' dominates 'i' if it is better or equal on all axes
            % and strictly better on at least one axis.
            if (all_x(j) <= all_x(i)) && (all_y(j) >= all_y(i)) && ...
               ((all_x(j) < all_x(i)) || (all_y(j) > all_y(i)))
                
                is_pareto(i) = false; % Point 'i' is dominated, so it's not on the front
                break; % No need to check other points, move to the next 'i'
            end
        end
    end

    % Extract the points that are on the Pareto front
    pareto_x = all_x(is_pareto);
    pareto_y = all_y(is_pareto);
    pareto_ids = {};
    if nargin > 2
       pareto_ids = all_ids(is_pareto); 
    end

    % --- Step 2: Plot the Results ---
    figure; % Create a new figure window
    hold on; % Allow multiple plots on the same axes

    % Plot all the models
    scatter(all_x, all_y, 50, 'b', 'o', 'DisplayName', 'Dominated Models'); % Blue circles for all models

    % Highlight the Pareto front models
    scatter(pareto_x, pareto_y, 100, 'r', 'o', 'filled', 'DisplayName', 'Pareto Front'); % Larger red filled circles

    % Sort the Pareto points to draw a connecting line
    [sorted_pareto_x, sort_idx] = sort(pareto_x);
    sorted_pareto_y = pareto_y(sort_idx);
    
    % Plot the connecting line
    plot(sorted_pareto_x, sorted_pareto_y, 'r-', 'LineWidth', 1.5, 'HandleVisibility', 'off'); % Red line

    % Add text labels to the Pareto points
    if ~isempty(pareto_ids)
        sorted_pareto_ids = pareto_ids(sort_idx);
        for k = 1:length(sorted_pareto_x)
            text(sorted_pareto_x(k) + 0.005, sorted_pareto_y(k), sorted_pareto_ids{k}, 'Color', 'red');
        end
    end
    
    % --- Step 3: Enhance the Plot ---
    title('Pareto Front for Model Selection');
    xlabel('Reconstruction Error (Lower is Better)');
    ylabel('Unified Cluster Quality Score (Higher is Better)');
    legend('show', 'Location', 'SouthEast');
    grid on;
    hold off;
end

function best_id = selectBestParetoPoint(all_x, all_y, all_ids)
    % selectBestParetoPoint: Identifies the Pareto front and selects the single
    % best compromise model using the Utopian Point distance method.
    
    % --- Step 1: Identify the Pareto Optimal Points (same as before) ---
    num_points = length(all_x);
    is_pareto = true(num_points, 1);
    for i = 1:num_points
        for j = 1:num_points
            if i == j, continue; end
            if (all_x(j) <= all_x(i)) && (all_y(j) >= all_y(i)) && ...
               ((all_x(j) < all_x(i)) || (all_y(j) > all_y(i)))
                is_pareto(i) = false;
                break;
            end
        end
    end
    pareto_x = all_x(is_pareto);
    pareto_y = all_y(is_pareto);
    pareto_ids = all_ids(is_pareto);

    % --- Step 2: Normalize the Pareto Points for Fair Distance Calculation ---
    % Normalize X-axis (Reconstruction Error) to [0, 1] where 0 is best
    min_x = min(pareto_x);
    max_x = max(pareto_x);
    norm_x = (pareto_x - min_x) / (max_x - min_x);

    % Normalize Y-axis (Cluster Score) to [0, 1] where 1 is best
    % This is already done if using the Unified Score, but we'll assume it's
    % on a [0,1] scale for this calculation.
    norm_y = pareto_y; % Assuming Y is already a [0,1] score

    % --- Step 3: Calculate Distance to the Utopian Point (0, 1) ---
    % The Utopian point in the normalized space is (x=0, y=1)
    % Distance formula: sqrt( (x2-x1)^2 + (y2-y1)^2 )
    distances = sqrt((norm_x - 0).^2 + (norm_y - 1).^2);

    % --- Step 4: Find the Point with the Minimum Distance ---
    [~, min_dist_idx] = min(distances);
    
    % Get the ID and coordinates of the winning model
    best_id = pareto_ids{min_dist_idx};
    best_x = pareto_x(min_dist_idx);
    best_y = pareto_y(min_dist_idx);
    
    % --- Step 5: Plotting (similar to before, but highlights the winner) ---
    figure;
    hold on;
    scatter(all_x, all_y, 50, 'b', 'o', 'DisplayName', 'Dominated Models');
    
    % Plot Pareto front points
    plot(pareto_x, pareto_y, 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'r', 'DisplayName', 'Pareto Front');

    % Highlight the single BEST point with a star
    scatter(best_x, best_y, 300, 'g', '*', 'LineWidth', 2, 'DisplayName', ['Best Model: ' best_id]);
    
    title('Pareto Front with Best Model Selection');
    xlabel('Reconstruction Error (Lower is Better)');
    ylabel('Unified Cluster Quality Score (Higher is Better)');
    legend('show', 'Location', 'SouthEast');
    grid on;
    hold off;
end
