function out = run_one_ng_scaling_size_exp2b(XMode,qModes,n,h,part,pre, ...
    directEnabled,taylorEnabled,directBlockSize,cfg)
%RUN_ONE_NG_SCALING_SIZE_EXP2B Stream one Yee size with independent budgets.
%
% The direct and Taylor methods have independent timeout states.  No full
% reconstructed field is stored.  Relative errors are accumulated blockwise
% only when both methods complete the entire size.

    NF = size(qModes,1);
    if numel(XMode) ~= 3*NF
        error('run_one_ng_scaling_size_exp2b:CoeffMismatch', ...
            'XMode must contain exactly 3*NF Fourier coefficients.');
    end
    if part.numPoints ~= (2*n+3)^3
        error('run_one_ng_scaling_size_exp2b:PartitionMismatch','Partition point count is inconsistent with n.');
    end

    coeff = {XMode(1:NF), XMode(NF+1:2*NF), XMode(2*NF+1:3*NF)};
    ranges = part.ranges;
    nCenters = size(ranges,1);
    groupSize = max(1,round(cfg.scaling.centersPerTimeoutCheck));
    timeout = cfg.scaling.timeoutSeconds;

    directElapsed = 0;
    taylorElapsed = 0;
    directRunning = logical(directEnabled);
    taylorRunning = logical(taylorEnabled);
    directTimedOut = false;
    taylorTimedOut = false;
    directCompletedOverBudget = false;
    taylorCompletedOverBudget = false;
    directProcessedLast = false;
    taylorProcessedLast = false;

    % Error accumulation remains valid only if both methods complete every
    % block of all three components.
    errorPossible = directRunning && taylorRunning;
    z = zeros(1,'like',qModes(1));
    err2sum = z;
    ref2sum = z;
    errInf = z;
    refInf = z;

    totalGroups = 3*ceil(nCenters/groupSize);
    groupCounter = 0;
    abortedAll = false;

    fprintf('    streaming %d centers/component in groups of <= %d ...\n',nCenters,groupSize);

    for ell = 1:3
        if abortedAll, break; end
        for s0 = 1:groupSize:nCenters
            s1 = min(nCenters,s0+groupSize-1);
            ids = s0:s1;
            groupCounter = groupCounter + 1;
            isLastOverall = (ell==3 && s1==nCenters);

            % -------------------------------------------------------------
            % Coordinate generation is outside both reconstruction timers.
            % -------------------------------------------------------------
            Rcells = cell(numel(ids),1);
            rcCells = cell(numel(ids),1);
            for ib = 1:numel(ids)
                [Rcells{ib},rcCells{ib}] = build_yee_block_points_exp2b( ...
                    ranges(ids(ib),:),h,ell,qModes(1));
            end
            wait_for_gpu();

            uT = {};
            uD = {};

            % Taylor first.  Warm-up is handled globally, so all timings
            % below correspond to production work only.
            if taylorRunning
                [uT,dtT] = reconstruct_taylor_group_exp2b( ...
                    coeff{ell},qModes,Rcells,rcCells,pre);
                taylorElapsed = taylorElapsed + dtT;
                if taylorElapsed >= timeout && ~isLastOverall
                    taylorTimedOut = true;
                    taylorRunning = false;
                    errorPossible = false;
                    fprintf('      Taylor reached %.1f s budget at component %d, centers %d:%d; stopping this and larger sizes.\n', ...
                        timeout,ell,s0,s1);
                elseif taylorElapsed >= timeout && isLastOverall
                    taylorCompletedOverBudget = true;
                end
                if isLastOverall
                    taylorProcessedLast = true;
                end
            end

            if directRunning
                [uD,dtD] = reconstruct_direct_group_exp2b( ...
                    coeff{ell},qModes,Rcells,directBlockSize);
                directElapsed = directElapsed + dtD;
                if directElapsed >= timeout && ~isLastOverall
                    directTimedOut = true;
                    directRunning = false;
                    errorPossible = false;
                    fprintf('      Direct reached %.1f s budget at component %d, centers %d:%d; stopping this and larger sizes.\n', ...
                        timeout,ell,s0,s1);
                elseif directElapsed >= timeout && isLastOverall
                    directCompletedOverBudget = true;
                end
                if isLastOverall
                    directProcessedLast = true;
                end
            end

            % -------------------------------------------------------------
            % Error is not part of either timing.  It is accumulated only
            % while both full-field reconstructions remain viable.
            % -------------------------------------------------------------
            if errorPossible && ~isempty(uT) && ~isempty(uD)
                for ib = 1:numel(ids)
                    d = uT{ib} - uD{ib};
                    err2sum = err2sum + sum(abs(d).^2);
                    ref2sum = ref2sum + sum(abs(uD{ib}).^2);
                    errInf = max(errInf,max(abs(d)));
                    refInf = max(refInf,max(abs(uD{ib})));
                end
            end

            clear uT uD Rcells rcCells

            if cfg.output.verbose
                reportEvery = max(1,round(totalGroups/20));
                if mod(groupCounter,reportEvery)==0 || isLastOverall
                    fprintf('      progress %5.1f%% | direct %.2f s | Taylor %.2f s\n', ...
                        100*groupCounter/totalGroups,directElapsed,taylorElapsed);
                end
            end

            if ~directRunning && ~taylorRunning
                abortedAll = true;
                break;
            end
        end
    end

    % Completion is unambiguous: a method must actually have processed the
    % final block of component 3.  This remains correct if the other method
    % timed out earlier and the surviving method continued alone.
    directCompleted = logical(directEnabled) && directProcessedLast && ~directTimedOut;
    taylorCompleted = logical(taylorEnabled) && taylorProcessedLast && ~taylorTimedOut;

    if ~directEnabled
        directStatus = 'skipped_after_budget';
        directTime = NaN;
    elseif directTimedOut
        directStatus = 'timeout_incomplete';
        directTime = NaN;
    elseif directCompletedOverBudget
        directStatus = 'completed_over_budget';
        directTime = directElapsed;
    elseif directCompleted
        directStatus = 'completed';
        directTime = directElapsed;
    else
        directStatus = 'incomplete';
        directTime = NaN;
    end

    if ~taylorEnabled
        taylorStatus = 'skipped_after_budget';
        taylorTime = NaN;
    elseif taylorTimedOut
        taylorStatus = 'timeout_incomplete';
        taylorTime = NaN;
    elseif taylorCompletedOverBudget
        taylorStatus = 'completed_over_budget';
        taylorTime = taylorElapsed;
    elseif taylorCompleted
        taylorStatus = 'completed';
        taylorTime = taylorElapsed;
    else
        taylorStatus = 'incomplete';
        taylorTime = NaN;
    end

    if directCompleted && taylorCompleted && errorPossible
        wait_for_gpu();
        e2num = sqrt(gather_scalar(err2sum));
        e2den = sqrt(gather_scalar(ref2sum));
        einfNum = gather_scalar(errInf);
        einfDen = gather_scalar(refInf);
        e2 = e2num/max(e2den,eps);
        eInf = einfNum/max(einfDen,eps);
    else
        e2 = NaN;
        eInf = NaN;
    end

    out = struct();
    out.n = n;
    out.Ng = part.numPoints;
    out.directTimeSeconds = directTime;
    out.directElapsedAtStopSeconds = directElapsed;
    out.directStatus = directStatus;
    out.directCompleted = directCompleted;
    out.directBudgetExceeded = directTimedOut || directCompletedOverBudget;
    out.taylorTimeSeconds = taylorTime;
    out.taylorElapsedAtStopSeconds = taylorElapsed;
    out.taylorStatus = taylorStatus;
    out.taylorCompleted = taylorCompleted;
    out.taylorBudgetExceeded = taylorTimedOut || taylorCompletedOverBudget;
    out.relativeError2 = e2;
    out.relativeErrorInf = eInf;
end
