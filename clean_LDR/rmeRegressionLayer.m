classdef rmeRegressionLayer < nnet.layer.RegressionLayer
    % Custom RMSE (Root Mean Square Error) regression layer
    % Designed for 1D time series data with 1000 time steps
    
    methods
        function loss = forwardLoss(~, Y, T)
            % Check input size
            sz = size(Y);
            
            % Expected input shape: [timeSteps × 1 × batchSize]
            if sz(1) ~= 1000
                error("Expected input time series length of 1000, but got %d", sz(1));
            end
            
            % Squared error per time step
            sqErr = (Y - T).^2;  % size: [1000 × 1 × batchSize]

            % Mean squared error per sample
            msePerSample = mean(sqErr, 1);  % [1 × 1 × batchSize]

            % Root mean error per sample
            rmsePerSample = sqrt(msePerSample);  % [1 × 1 × batchSize]

            % Final loss = mean across batch
            loss = mean(rmsePerSample, 'all');
        end
    end
end
