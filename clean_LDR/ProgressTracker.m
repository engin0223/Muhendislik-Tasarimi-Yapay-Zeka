classdef ProgressTracker < handle
    properties (Access = private)
        IterCount = 0
        StartTime
        NumWindows
        DataQueue
        LogInterval
    end

    methods
        function obj = ProgressTracker(n_windows, log_interval_rate)
            % Constructor: Set the number of windows and optional log interval rate
            if nargin < 2
                log_interval_rate = 0.01;  % Default value
            end
            % Constructor: Set the number of windows
            obj.NumWindows = n_windows;
            obj.LogInterval = log_interval_rate;
            obj.DataQueue = parallel.pool.DataQueue;
            afterEach(obj.DataQueue, @(x) obj.update(x)); % Hook for parallel updates
        end

        function start(obj)
            % Start the timer and reset counter
            obj.IterCount = 0;
            obj.StartTime = tic;
        end

        function reset(obj)
            % Reset the counter (manual call)
            obj.IterCount = 0;
        end

        function update(obj, x)
            % Increment iteration and optionally display progress
            for i=1:x
                obj.IterCount = obj.IterCount + 1;
                if mod(obj.IterCount, floor(obj.NumWindows*obj.LogInterval)) == 0
                    elapsed = toc(obj.StartTime);
                    progress = obj.IterCount / obj.NumWindows;
                    remaining = elapsed / progress * (1 - progress);
                    fprintf('Progress: %.2f%% | Elapsed: %s | Remaining: %s\n', ...
                        progress * 100, obj.formatTime(elapsed), obj.formatTime(remaining));
                end
            end
        end

        function dq = getQueue(obj)
            % Return the DataQueue object for use in parfor
            dq = obj.DataQueue;
        end
    end
    
    methods (Access = private)
        function str = formatTime(~, seconds)
            % Format time as hh:mm:ss
            hours = floor(seconds / 3600);
            minutes = floor(mod(seconds, 3600) / 60);
            secs = mod(seconds, 60);
            str = sprintf('%02d:%02d:%05.2f', hours, minutes, secs);
        end
    end
end
