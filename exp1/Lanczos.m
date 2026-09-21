function [ev,ew,iter,info] = Lanczos(fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose,monitorEveryIteration,monitorStride)
%LANCZOS Restarted Krylov-Schur/Lanczos solver for a Hermitian GEP.
%
% The Krylov recurrence, restart strategy, GEP Gram-Schmidt, Ritz extraction,
% and convergence test are the user-supplied algorithm.  Two optional
% interface/diagnostic additions are provided:
%   VERBOSE                 controls per-iteration printing;
%   MONITOREVERYITERATION   enables diagnostic Ritz-residual/time traces;
%   MONITORSTRIDE           records a diagnostic point every MONITORSTRIDE
%                           outer Lanczos steps.  These diagnostics do not
%                           change Krylov vectors, restart decisions, or the
%                           convergence test.

    if nargin < 12 || isempty(verbose)
        verbose = true;
    end
    if nargin < 13 || isempty(monitorEveryIteration)
        monitorEveryIteration = false;
    end
    if nargin < 14 || isempty(monitorStride)
        monitorStride = 1;
    end
    monitorStride = max(1,round(monitorStride));

    if flag_gpu && ~isa(v0,'gpuArray')
        v0 = gpuArray(v0);
    end

    nrm0 = sqrt(abs(gather_scalar(v0'*fncB(v0))));
    v0 = v0/nrm0;
    [ew,ev,iter,~,info] = Lanczos_iteration( ...
        fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose,monitorEveryIteration,monitorStride);
end

function [d,U,kk,res,info] = Lanczos_iteration( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v,flag_gpu,verbose,monitorEveryIteration,monitorStride)

    eps0 = eps('double');
    Q = zeros(n,p,'like',v);
    k0 = k;

    Alpha = zeros(0,1);
    Beta = zeros(0,1);
    beta = 0;
    d = zeros(0,1);
    c = zeros(0,1);
    justRestarted = false;
    sizeV = 1;
    kk = 0;

    info.restart = [];
    info.totalIter = [];
    info.maxWantedResidual = [];
    info.meanWantedResidual = [];
    info.ritzValues = {};
    info.ritzResiduals = {};
    info.traceIter = [];
    info.traceMaxWantedResidual = [];
    info.traceMeanWantedResidual = [];
    info.traceTimeSeconds = [];
    info.diagnosticTimeSeconds = 0;
    % Convergence-status fields. These do not alter the Lanczos recurrence,
    % restart selection, projected eigensolves, or stopping thresholds.
    info.converged = false;
    info.numConverged = 0;
    info.hitMaxRestart = false;
    info.finalRestart = 0;

    Usmall = [];
    ind = [];
    res = [];
    Num_iter = 0;

    sync_device(flag_gpu);
    runTimer = tic;

    for mm = 1:maxiter
        for jj = sizeV:p
            iterTimer = tic;
            Q(:,jj) = v;
            Aq = fncA(Q(:,jj));
            alpha = real(gather_scalar(Aq'*Q(:,jj)));
            pr = fncBinv(Aq);

            if jj==1
                r = pr-alpha*Q(:,jj);
            elseif justRestarted
                r = GramSchmitGEP(Q,pr,fncB,jj);
                justRestarted = false;
            else
                r = pr-alpha*Q(:,jj)-beta*Q(:,jj-1);
                corr = gather_scalar(fncB(Q(:,jj-1))'*r);
                r = r-corr*Q(:,jj-1);
            end

            corr = gather_scalar(fncB(Q(:,jj))'*r);
            r = r-corr*Q(:,jj);
            r = GramSchmitGEP(Q,r,fncB,jj);
            r = GramSchmitGEP(Q,r,fncB,jj);

            beta2 = real(gather_scalar(r'*fncB(r)));
            if beta2<=0 || sqrt(beta2)<1e-14
                warning('Lanczos breakdown or near breakdown at step %d.',kk);
                beta = sqrt(max(beta2,0));
                break;
            end

            beta = sqrt(beta2);
            v = r/beta;
            Alpha(end+1,1) = alpha; %#ok<AGROW>
            Beta(end+1,1) = beta; %#ok<AGROW>
            kk = kk+1;
            Num_iter = Num_iter + 1;

            if verbose
                sync_device(flag_gpu);
                fprintf('Lanczos iter num %d, spend time %.2f. \n', Num_iter, toc(iterTimer));
            end

            % Diagnostic-only trace.  The projected eigendecomposition here
            % is not used by the recurrence, restart, or stopping test.
            if monitorEveryIteration && kk>=k0 && mod(kk,monitorStride)==0
                diagTimer = tic;
                [~,~,resTrace] = projected_ritz_data(d,c,k,Alpha,Beta,beta,k0);
                sync_device(flag_gpu);
                diagElapsed = toc(diagTimer);
                info.diagnosticTimeSeconds = info.diagnosticTimeSeconds + diagElapsed;

                info.traceIter(end+1,1) = kk; %#ok<AGROW>
                info.traceMaxWantedResidual(end+1,1) = max(resTrace); %#ok<AGROW>
                info.traceMeanWantedResidual(end+1,1) = mean(resTrace); %#ok<AGROW>
                info.traceTimeSeconds(end+1,1) = ...
                    toc(runTimer)-info.diagnosticTimeSeconds; %#ok<AGROW>
            end
        end

        H1 = diag(Alpha)+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
        H = blkdiag(diag(d),H1);
        if ~isempty(d)
            H(1:k,k+1) = c';
            H(k+1,1:k) = c;
        end

        Alpha = zeros(0,1);
        Beta = zeros(0,1);
        [Usmall,d] = eig(H,'vector');
        res = abs(beta*Usmall(end,:));
        [~,ind] = sort(real(d),'descend');
        d = d(ind);
        res = res(ind);

        wanted = 1:min(k0,numel(res));
        info.restart(end+1) = mm;
        info.totalIter(end+1) = kk;
        info.maxWantedResidual(end+1) = max(res(wanted));
        info.meanWantedResidual(end+1) = mean(res(wanted));
        info.ritzValues{end+1} = d(wanted);
        info.ritzResiduals{end+1} = res(wanted);

        % Always retain the restart/final point in the diagnostic trace,
        % even when it is not a multiple of monitorStride.  The projected
        % eigendecomposition above is part of the original algorithm, so no
        % extra diagnostic timing is introduced here.
        if monitorEveryIteration
            traceTime = toc(runTimer)-info.diagnosticTimeSeconds;
            if isempty(info.traceIter) || info.traceIter(end)~=kk
                info.traceIter(end+1,1) = kk; %#ok<AGROW>
                info.traceMaxWantedResidual(end+1,1) = max(res(wanted)); %#ok<AGROW>
                info.traceMeanWantedResidual(end+1,1) = mean(res(wanted)); %#ok<AGROW>
                info.traceTimeSeconds(end+1,1) = traceTime; %#ok<AGROW>
            else
                info.traceMaxWantedResidual(end,1) = max(res(wanted));
                info.traceMeanWantedResidual(end,1) = mean(res(wanted));
                info.traceTimeSeconds(end,1) = traceTime;
            end
        end

        wantedRes = res(wanted).';
        wantedD = d(wanted);
        threshold = tol*max(eps0^(2/3),abs(wantedD));
        nconv = nnz(wantedRes<threshold);
        info.numConverged = nconv;
        info.finalRestart = mm;
        if nconv>=k0
            info.converged = true;
            break;
        elseif mm==maxiter
            info.hitMaxRestart = true;
            break;
        end

        k = k0+min(nconv,floor((p-k0)/2));
        if k==1 && p>3
            k = floor(p/2);
        end

        keep = ind(1:k);
        Ukeep = Usmall(:,keep);
        if flag_gpu
            Q(:,1:k) = Q*gpuArray(cast(Ukeep,classUnderlying(Q)));
        else
            Q(:,1:k) = Q*Ukeep;
        end
        d = d(1:k);
        c = beta*Ukeep(end,:);
        justRestarted = true;
        sizeV = k+1;
    end

    Ukeep = Usmall(:,ind(1:k0));
    d = d(1:k0);
    if flag_gpu
        U = Q*gpuArray(cast(Ukeep,classUnderlying(Q)));
    else
        U = Q*Ukeep;
    end

    sync_device(flag_gpu);
    info.wallTimeIncludingDiagnostics = toc(runTimer);
    info.wallTimeExcludingDiagnostics = max( ...
        info.wallTimeIncludingDiagnostics-info.diagnosticTimeSeconds,0);
end

function [dWanted,Uall,resWanted] = projected_ritz_data(d,c,k,Alpha,Beta,beta,k0)
% Diagnostic-only Ritz data for the current projected problem.
    H1 = diag(Alpha)+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
    H = blkdiag(diag(d),H1);
    if ~isempty(d)
        H(1:k,k+1) = c';
        H(k+1,1:k) = c;
    end
    [Uall,dAll] = eig(H,'vector');
    resAll = abs(beta*Uall(end,:));
    [~,idx] = sort(real(dAll),'descend');
    wanted = idx(1:min(k0,numel(idx)));
    dWanted = dAll(wanted);
    resWanted = resAll(wanted).';
end

function r = GramSchmitGEP(V,r,fncB,jj)
    for kk = 1:jj
        tmp = gather_scalar(fncB(V(:,kk))'*r);
        r = r-tmp*V(:,kk);
    end
end

function sync_device(flag_gpu)
    if flag_gpu
        wait(gpuDevice);
    end
end
