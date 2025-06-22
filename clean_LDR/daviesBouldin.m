function DB = daviesBouldin(X, labels)
%DAVIESBOULDIN Calculate the Davies-Bouldin Index for clustering validation.
%   DB = DAVIESBOULDIN(X, labels) computes the Davies-Bouldin Index (DB)
%   for a given dataset X and its corresponding cluster labels.
%
%   Inputs:
%       X      : N-by-D data matrix, where N is the number of data points
%                and D is the dimensionality of the data.
%       labels : N-by-1 vector of cluster assignments for each data point.
%                Cluster labels should be positive integers.
%
%   Output:
%       DB     : The Davies-Bouldin Index value. Lower values indicate
%                better clustering.
%
%   Reference:
%       Davies, David L.; Bouldin, Donald W. (1979). "A Cluster Separation
%       Measure". IEEE Transactions on Pattern Analysis and Machine
%       Intelligence. PAMI-1 (2): 224-227.
%       https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index

% Check for minimum number of clusters
uniqueLabels = unique(labels);
numClusters = length(uniqueLabels);

if numClusters < 2
    error('Davies-Bouldin Index requires at least two clusters.');
end

% 1. Calculate Cluster Centroids (mi)
% m_i = (1/|C_i|) * sum_{x in C_i} x
centroids = zeros(numClusters, size(X, 2));
for k = 1:numClusters
    cluster_k_points = X(labels == uniqueLabels(k), :);
    centroids(k, :) = mean(cluster_k_points, 1);
end

% 2. Calculate Intra-cluster Scatter (si)
% s_i = (1/|C_i|) * sum_{x in C_i} ||x - m_i||_2
scatter = zeros(numClusters, 1);
for k = 1:numClusters
    cluster_k_points = X(labels == uniqueLabels(k), :);
    numPoints_k = size(cluster_k_points, 1);
    if numPoints_k == 0
        scatter(k) = 0; % Handle empty clusters if any (though typically labels imply non-empty)
        continue;
    end
    diff_from_centroid = cluster_k_points - repmat(centroids(k, :), numPoints_k, 1);
    scatter(k) = sum(sqrt(sum(diff_from_centroid.^2, 2))) / numPoints_k;
end

% 3. Calculate Inter-cluster Distance (d_ij)
% d_ij = ||m_i - m_j||_2
distance_centroids = pdist2(centroids, centroids, 'euclidean'); % Euclidean distance between centroids

% 4. Calculate Similarity Measure (R_ij)
% R_ij = (s_i + s_j) / d_ij
R = zeros(numClusters, numClusters);
for i = 1:numClusters
    for j = 1:numClusters
        if i == j
            R(i, j) = 0; % R_ii is not defined in the formula, set to 0 or inf to be ignored
        else
            if distance_centroids(i, j) == 0 % Avoid division by zero for coincident centroids
                R(i, j) = inf;
            else
                R(i, j) = (scatter(i) + scatter(j)) / distance_centroids(i, j);
            end
        end
    end
end

% 5. Calculate Di
% D_i = max_{j!=i} R_ij
D = zeros(numClusters, 1);
for i = 1:numClusters
    % Find maximum R_ij for j != i
    tempR = R(i, :);
    tempR(i) = -inf; % Exclude R_ii from max calculation
    D(i) = max(tempR);
end

% Handle cases where max might be -inf if all other clusters are coincident
% with current one (should ideally be caught by distance_centroids == 0 check)
D(isinf(D)) = 0; % If max becomes inf due to division by zero, it's a very bad clustering.
                % For practical purposes, if a cluster has coincident centroids
                % with ALL other clusters, this should be handled.
                % A value of 0 might be appropriate if all R_ij are 0 or negative.
                % Or, ideally, the inf would propagate to the sum, making DB inf.

% 6. Calculate Davies-Bouldin Index
% DB = (1/k) * sum_{i=1 to k} D_i
DB = sum(D) / numClusters;

end
