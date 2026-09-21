function report = check_experiment2b_results(results,cfg,throwOnFailure)
%CHECK_EXPERIMENT2B_RESULTS Structural and numerical consistency checks.
    if nargin<3, throwOnFailure=false; end
    T=results.scaling;
    failures={}; warningsList={};

    if height(T)~=numel(cfg.scaling.nValues)
        failures{end+1}='Scaling table row count does not match configured nValues.'; %#ok<AGROW>
    end
    if any(T.n(:)~=cfg.scaling.nValues(:))
        failures{end+1}='Scaling table n-values differ from configuration.'; %#ok<AGROW>
    end
    expectedNg=(2*T.n+3).^3;
    if any(T.Ng~=expectedNg)
        failures{end+1}='At least one N_G value is inconsistent with (2n+3)^3.'; %#ok<AGROW>
    end
    if any(diff(T.Ng)<=0)
        failures{end+1}='N_G must be strictly increasing.'; %#ok<AGROW>
    end

    tol=cfg.scaling.radiusRelativeTolerance;
    if any(T.RhoMax > T.RhoTarget.*(1+tol))
        failures{end+1}='A Taylor partition violates the adopted phase-radius threshold.'; %#ok<AGROW>
    end
    ib=find(T.n==cfg.scaling.baselineN,1);
    if isempty(ib) || T.BlocksPerDimension(ib)~=cfg.scaling.baselineBlocksPerDimension
        failures{end+1}='Baseline n=40 does not reproduce the adopted k=4 partition.'; %#ok<AGROW>
    end
    for i=1:height(T)
        q=choose_partition_for_radius_exp2b(T.n(i),cfg.scaling.h,results.qL1Max, ...
            results.rhoTarget,cfg.scaling.radiusRelativeTolerance);
        if q.blocksPerDimension~=T.BlocksPerDimension(i) || q.numCenters~=T.NumCenters(i)
            failures{end+1}=sprintf('Partition at n=%d is not the minimal radius-admissible partition.',T.n(i)); %#ok<AGROW>
        end
    end

    rows=find(T.RowComplete);
    for ii=1:numel(rows)
        i=rows(ii);
        ds=string(T.DirectStatus(i)); ts=string(T.TaylorStatus(i));
        if ds=="completed" || ds=="completed_over_budget"
            if ~(isfinite(T.DirectTimeSeconds(i)) && T.DirectTimeSeconds(i)>0 && T.DirectCompleted(i))
                failures{end+1}=sprintf('Invalid completed direct timing at n=%d.',T.n(i)); %#ok<AGROW>
            end
        elseif ds=="timeout_incomplete"
            if isfinite(T.DirectTimeSeconds(i)) || T.DirectElapsedAtStopSeconds(i)<0.95*cfg.scaling.timeoutSeconds
                failures{end+1}=sprintf('Invalid direct timeout bookkeeping at n=%d.',T.n(i)); %#ok<AGROW>
            end
        elseif ds~="skipped_after_budget"
            failures{end+1}=sprintf('Unexpected direct status "%s" at completed n=%d.',ds,T.n(i)); %#ok<AGROW>
        end

        if ts=="completed" || ts=="completed_over_budget"
            if ~(isfinite(T.TaylorTimeSeconds(i)) && T.TaylorTimeSeconds(i)>0 && T.TaylorCompleted(i))
                failures{end+1}=sprintf('Invalid completed Taylor timing at n=%d.',T.n(i)); %#ok<AGROW>
            end
        elseif ts=="timeout_incomplete"
            if isfinite(T.TaylorTimeSeconds(i)) || T.TaylorElapsedAtStopSeconds(i)<0.95*cfg.scaling.timeoutSeconds
                failures{end+1}=sprintf('Invalid Taylor timeout bookkeeping at n=%d.',T.n(i)); %#ok<AGROW>
            end
        elseif ts~="skipped_after_budget"
            failures{end+1}=sprintf('Unexpected Taylor status "%s" at completed n=%d.',ts,T.n(i)); %#ok<AGROW>
        end

        if isfinite(T.RelativeFieldError2(i))
            if ~(T.DirectCompleted(i) && T.TaylorCompleted(i))
                failures{end+1}=sprintf('e2 exists without two complete methods at n=%d.',T.n(i)); %#ok<AGROW>
            end
            if ~isfinite(T.RelativeFieldErrorInf(i)) || T.RelativeFieldError2(i)<0 || T.RelativeFieldErrorInf(i)<0
                failures{end+1}=sprintf('Invalid error diagnostic at n=%d.',T.n(i)); %#ok<AGROW>
            end
            if T.RelativeFieldError2(i)>1e-4
                warningsList{end+1}=sprintf('e2=%.3e at n=%d is larger than expected for the adopted p=10 radius regime.', ...
                    T.RelativeFieldError2(i),T.n(i)); %#ok<AGROW>
            end
        else
            if T.DirectCompleted(i) && T.TaylorCompleted(i)
                failures{end+1}=sprintf('Missing e2 despite two complete methods at n=%d.',T.n(i)); %#ok<AGROW>
            end
        end
    end

    % Once a method exhausts its budget, it must never reappear at a larger
    % completed row.  This catches accidental state reactivation.
    for method={"Direct","Taylor"}
        m=method{1};
        if m=="Direct"
            exceeded=T.DirectBudgetExceeded; completed=T.DirectCompleted;
        else
            exceeded=T.TaylorBudgetExceeded; completed=T.TaylorCompleted;
        end
        first=find(exceeded & T.RowComplete,1);
        if ~isempty(first)
            later=(first+1):height(T);
            if any(completed(later) & T.RowComplete(later))
                failures{end+1}=sprintf('%s method reactivated after exhausting its budget.',m); %#ok<AGROW>
            end
        end
    end

    report=struct('passed',isempty(failures),'failures',{failures},'warnings',{warningsList});
    if cfg.output.verbose || ~isempty(failures)
        fprintf('\n[Result self-check] %d failure(s), %d warning(s).\n',numel(failures),numel(warningsList));
        for i=1:numel(failures), fprintf('  FAILURE: %s\n',failures{i}); end
        for i=1:numel(warningsList), fprintf('  WARNING: %s\n',warningsList{i}); end
    end
    if throwOnFailure && ~isempty(failures)
        error('check_experiment2b_results:Failed','Experiment 2(b) result self-check failed; see messages above.');
    end
end
