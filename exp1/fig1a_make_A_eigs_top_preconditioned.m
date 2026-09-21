function [Top,getStats,meta] = fig1a_make_A_eigs_top_preconditioned(sys,cfg,epsMean)
%MAKE_A_EIGS_TOP_PRECONDITIONED Matrix-free Method-A transformed operator.
%
%   T(x) = (K - sigma*M)^(-1) M*x.
%
% Outer eigensolver
% -----------------
% MATLAB eigs manages the Krylov-Schur process on the CPU.
%
% Inner shifted solve
% -------------------
% Each Top application transfers the CPU vector to the GPU and solves
%
%   (K-sigma*M)y = M*x
%
% by MATLAB MINRES with the spectral SPD preconditioner
%
%   P_A = K + tau*I,  tau = sigma*mean(epsilon).
%
% P_A^{-1} is applied exactly mode by mode using the Fourier curl-curl
% symbol. No LU/LDL factorization is formed.
%
% Robust MINRES handling
% ----------------------
% Each shifted system is solved by uninterrupted MINRES segments. Normally
% one segment converges. If MATLAB returns flag=3 (stagnation: two
% consecutive iterates are identical), the true residual is recomputed.
% If the requested tolerance is not yet satisfied, a restart is allowed only
% when the segment produced new iterations and reduced the true residual.
% Zero-iteration or zero-progress stagnation switches to a residual-correction
% solve A*delta=r instead of repeating the same restart. Guard counters make
% every recovery path finite. Iterations and times from all MINRES calls,
% including residual corrections, are accumulated.  If repeated recovery
% cycles reach a true-residual plateau below the configured 1e-9 ceiling,
% the current shifted solve is explicitly marked PLATEAU ACCEPTED and returned.
%
% The expected MATLAB warning that the requested tolerance may be
% unattainable is suppressed inside MINRES; the code reports flag, true
% residual, restart count, and cumulative work explicitly instead.
%
% Work counts
% -----------
% topVectorApplications is the primary outer work count.
% totalMINRESIterations is the cumulative number of preconditioned MINRES
% iterations over every shifted system requested by eigs, including all
% stagnation/maxit restarts and residual-correction solves.


    verbose = isfield(cfg,'runtime') && isfield(cfg.runtime,'verboseInner') && cfg.runtime.verboseInner;

    sigma = cfg.shift.sigma;
    tol = cfg.shift.minresTolerance;
    maxit = cfg.shift.minresMaxit;
    tau = sigma*epsMean;

    Ashift = @(xg) sys.Kop(xg) - sigma*sys.mass.Mop(xg);
    [Pinv,pmeta] = build_original_shift_preconditioner_A_only(sys,tau);

    stats = init_stats();
    Top = @applyT;
    getStats = @getStatsLocal;
    meta = struct( ...
        'sigma',sigma, ...
        'tolMINRES',tol, ...
        'maxitMINRESPerSegment',maxit, ...
        'preconditioner',pmeta, ...
        'outerExecution','CPU MATLAB eigs / Krylov-Schur', ...
        'innerExecution','GPU K/M + GPU preconditioned MINRES with guarded restart/residual correction + recorded plateau fallback');

    function Ycpu = applyT(Xcpu)
        if size(Xcpu,1) ~= sys.ndof
            error('Top input must have sys.ndof rows.');
        end
        if isa(Xcpu,'gpuArray')
            stats.unexpectedGPUInputCalls = stats.unexpectedGPUInputCalls + 1;
            error(['Method A expects MATLAB eigs Krylov vectors on the CPU, ', ...
                   'but Top received a gpuArray input.']);
        end
        if ~isnumeric(Xcpu)
            error('Top input from eigs must be a numeric CPU array.');
        end

        nrhs = size(Xcpu,2);
        firstOuterIndex = stats.topVectorApplications + 1;
        stats.topFunctionCalls = stats.topFunctionCalls + 1;
        stats.topVectorApplications = stats.topVectorApplications + nrhs;

        wait(gpuDevice);
        topTimer = tic;

        h2dTimer = tic;
        Xg = to_gpu(Xcpu,cfg.gpu.precision);
        wait_for_gpu(Xg);
        h2dSeconds = toc(h2dTimer);

        Yg = complex(zeros(size(Xg),'like',Xg));

        for col = 1:nrhs
            outerIndex = firstOuterIndex + col - 1;
            xg = Xg(:,col);

            rhsTimer = tic;
            rhs = sys.mass.Mop(xg);
            wait_for_gpu(rhs);
            rhsSeconds = toc(rhsTimer);

            [yg,solveInfo] = solve_shifted_system(rhs,outerIndex);
            Yg(:,col) = yg;

            stats.totalMINRESSolves = stats.totalMINRESSolves + 1;
            stats.totalMINRESCalls = stats.totalMINRESCalls + solveInfo.segmentCount;
            stats.totalMINRESIterations = stats.totalMINRESIterations + solveInfo.totalIterations;
            stats.totalMINRESTime = stats.totalMINRESTime + solveInfo.solveSeconds;
            stats.totalResidualCheckTime = stats.totalResidualCheckTime + solveInfo.residualCheckSeconds;
            stats.totalRHSMapTime = stats.totalRHSMapTime + rhsSeconds;
            stats.totalStagnationRestarts = stats.totalStagnationRestarts + solveInfo.stagnationRestarts;
            stats.totalMaxitRestarts = stats.totalMaxitRestarts + solveInfo.maxitRestarts;
            stats.totalResidualCorrections = stats.totalResidualCorrections + solveInfo.residualCorrections;
            stats.totalCorrectionMINRESCalls = stats.totalCorrectionMINRESCalls + solveInfo.correctionMINRESCalls;
            stats.totalCorrectionMINRESIterations = stats.totalCorrectionMINRESIterations + solveInfo.correctionMINRESIterations;
            stats.totalZeroProgressEvents = stats.totalZeroProgressEvents + solveInfo.zeroProgressEvents;
            stats.totalRecoveryEvents = stats.totalRecoveryEvents + solveInfo.recoveryEvents;
            stats.totalPlateauAccepts = stats.totalPlateauAccepts + double(solveInfo.plateauAccepted);
            if solveInfo.plateauAccepted
                stats.maxPlateauAcceptedResidual = max(stats.maxPlateauAcceptedResidual,solveInfo.plateauAcceptedResidual);
                stats.plateauAcceptedResiduals(end+1,1) = solveInfo.plateauAcceptedResidual; %#ok<AGROW>
            end
            stats.maxMINRESIterations = max(stats.maxMINRESIterations,solveInfo.totalIterations);
            stats.maxMINRESRelativeResidual = max( ...
                stats.maxMINRESRelativeResidual,solveInfo.finalExplicitRelres);

            stats.flags(end+1,1) = solveInfo.finalFlag; %#ok<AGROW>
            stats.iterations(end+1,1) = solveInfo.totalIterations; %#ok<AGROW>
            stats.relativeResiduals(end+1,1) = solveInfo.finalReportedRelres; %#ok<AGROW>
            stats.explicitRelativeResiduals(end+1,1) = solveInfo.finalExplicitRelres; %#ok<AGROW>
            stats.lastResidualNorms(end+1,1) = solveInfo.lastResidualNorm; %#ok<AGROW>
            stats.solveTimes(end+1,1) = solveInfo.solveSeconds; %#ok<AGROW>
            stats.residualCheckTimes(end+1,1) = solveInfo.residualCheckSeconds; %#ok<AGROW>
            stats.rhsTimes(end+1,1) = rhsSeconds; %#ok<AGROW>
            stats.segmentCounts(end+1,1) = solveInfo.segmentCount; %#ok<AGROW>
            stats.stagnationRestartsPerTop(end+1,1) = solveInfo.stagnationRestarts; %#ok<AGROW>
            stats.maxitRestartsPerTop(end+1,1) = solveInfo.maxitRestarts; %#ok<AGROW>
            stats.residualCorrectionsPerTop(end+1,1) = solveInfo.residualCorrections; %#ok<AGROW>
            stats.correctionMINRESIterationsPerTop(end+1,1) = solveInfo.correctionMINRESIterations; %#ok<AGROW>
            stats.zeroProgressEventsPerTop(end+1,1) = solveInfo.zeroProgressEvents; %#ok<AGROW>
            stats.recoveryEventsPerTop(end+1,1) = solveInfo.recoveryEvents; %#ok<AGROW>
            stats.plateauAcceptedPerTop(end+1,1) = double(solveInfo.plateauAccepted); %#ok<AGROW>
            stats.plateauAcceptedResidualPerTop(end+1,1) = solveInfo.plateauAcceptedResidual; %#ok<AGROW>

            if solveInfo.plateauAccepted
                statusText = 'PLATEAU ACCEPTED';
            else
                statusText = 'CONVERGED';
            end
            vprintf(['[Outer Top %d] preconditioned MINRES %s: ', ...
                     'iter=%d, true relres=%.3e, segments=%d, ', ...
                     'stagnation restarts=%d, corrections=%d, solve=%.3f s, ', ...
                     'cumulative MINRES=%d\n'], ...
                    outerIndex,statusText,solveInfo.totalIterations,solveInfo.finalExplicitRelres, ...
                    solveInfo.segmentCount,solveInfo.stagnationRestarts,solveInfo.residualCorrections, ...
                    solveInfo.solveSeconds,stats.totalMINRESIterations);
        end

        wait_for_gpu(Yg);
        d2hTimer = tic;
        Ycpu = gather(Yg);
        d2hSeconds = toc(d2hTimer);

        topSeconds = toc(topTimer);
        stats.totalH2DTime = stats.totalH2DTime + h2dSeconds;
        stats.totalD2HTime = stats.totalD2HTime + d2hSeconds;
        stats.totalTopTime = stats.totalTopTime + topSeconds;
        stats.h2dTimes(end+1,1) = h2dSeconds; %#ok<AGROW>
        stats.d2hTimes(end+1,1) = d2hSeconds; %#ok<AGROW>
        stats.topTimes(end+1,1) = topSeconds; %#ok<AGROW>
    end

    function [yg,info] = solve_shifted_system(rhs,outerIndex)
        % One shifted system with guarded recovery.
        %
        % Normal path:
        %   Solve A*y = rhs by preconditioned MINRES.
        %
        % Recovery path:
        %   - flag=3 with positive iteration count and real residual decrease:
        %       restart from the current iterate.
        %   - zero-iteration stagnation OR a nonzero segment that fails to
        %       reduce the true residual:
        %       switch to residual correction
        %           A*delta = r,   r = rhs - A*y,
        %       then update y <- y + delta.
        %
        % Every loop branch must either (i) accept a converged solution,
        % (ii) increase the accumulated MINRES iteration count, or
        % (iii) consume a bounded recovery event.  This prevents the former
        % zero-iteration restart dead loop.

        x0 = zeros(size(rhs),'like',rhs);
        rhsNorm = double(gather_scalar(norm(rhs,2)));
        if ~(isfinite(rhsNorm) && rhsNorm > 0)
            error('Invalid shifted-system RHS norm in Outer Top %d.',outerIndex);
        end

        totalIterations = 0;
        segmentCount = 0;
        stagnationRestarts = 0;
        maxitRestarts = 0;
        residualCorrections = 0;
        correctionMINRESCalls = 0;
        correctionMINRESIterations = 0;
        zeroProgressEvents = 0;
        recoveryEvents = 0;
        plateauAccepted = false;
        plateauAcceptedResidual = NaN;
        plateauNoProgressEvents = 0;
        solveSeconds = 0;
        residualCheckSeconds = 0;
        finalFlag = NaN;
        finalReportedRelres = NaN;
        finalExplicitRelres = Inf;
        lastResidualNorm = NaN;
        previousExplicitRelres = Inf;
        yg = x0;

        maxRecoveryEvents = cfg.shift.maxRecoveryEventsPerTop;
        maxCorrectionCycles = cfg.shift.maxResidualCorrectionCycles;
        maxZeroProgressEvents = cfg.shift.maxZeroProgressEvents;
        minProgressFraction = cfg.shift.minResidualProgressFraction;
        correctionSafety = cfg.shift.correctionSafetyFactor;
        correctionMaxTol = cfg.shift.correctionMaxRelativeTolerance;
        correctionMinTol = cfg.shift.correctionMinRelativeTolerance;
        plateauCeiling = cfg.shift.plateauAcceptanceCeiling;
        plateauMinEvents = cfg.shift.plateauMinFailedRecoveryCycles;
        plateauProgressFraction = cfg.shift.plateauMinRelativeReduction;

        % The requested MINRES tolerance remains tol=1e-10.  A relaxed
        % plateau acceptance is permitted only after repeated recovery cycles
        % fail to reduce the TRUE residual materially and only when that
        % residual is already below the separately configured ceiling (1e-9
        % in the formal Fig. 1(a) experiment).  Thus the normal stopping
        % criterion is never silently changed to 1e-9.

        while true
            if recoveryEvents > maxRecoveryEvents
                if plateauNoProgressEvents >= plateauMinEvents && ...
                        finalExplicitRelres <= plateauCeiling
                    plateauAccepted = true;
                    plateauAcceptedResidual = finalExplicitRelres;
                    fprintf(['  [Outer Top %d] PLATEAU ACCEPTED at recovery guard: ', ...
                             'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                            outerIndex,finalExplicitRelres,plateauCeiling,plateauNoProgressEvents);
                    break;
                end
                error(['Recovery-event guard triggered in Outer Top %d after %d events. ', ...
                       'Current true relres=%.3e. This is a controlled stop to prevent ', ...
                       'an infinite restart/recovery loop.'], ...
                      outerIndex,recoveryEvents,finalExplicitRelres);
            end

            segmentCount = segmentCount + 1;

            warnState = warning;
            warnCleanup = onCleanup(@()warning(warnState)); %#ok<NASGU>
            warning('off','all');

            segmentMaxit = maxit;
            segmentTimer = tic;
            [yNew,flag,relres,iter,resvec] = minres( ...
                Ashift,rhs,tol,segmentMaxit,Pinv,[],x0);
            wait_for_gpu(yNew);
            segmentSeconds = toc(segmentTimer);

            warning(warnState);
            clear warnCleanup

            if ~isa(yNew,'gpuArray')
                error('MINRES returned a non-gpuArray solution in Outer Top %d.',outerIndex);
            end

            flagCPU = double(gather_scalar(flag));
            relresCPU = double(gather_scalar(relres));
            iterCPU = double(gather_scalar(iter));
            if isempty(resvec)
                lastRes = NaN;
            else
                lastRes = double(gather_scalar(resvec(end)));
            end

            if any(~isfinite([flagCPU,relresCPU,iterCPU]))
                error('MINRES produced nonfinite scalar statistics in Outer Top %d.',outerIndex);
            end

            totalIterations = totalIterations + iterCPU;
            solveSeconds = solveSeconds + segmentSeconds;

            if flagCPU == 0
                explicitRelres = relresCPU;
            else
                [explicitRelres,~,checkSeconds] = explicit_relative_residual(yNew,rhs,rhsNorm);
                residualCheckSeconds = residualCheckSeconds + checkSeconds;
            end

            finalFlag = flagCPU;
            finalReportedRelres = relresCPU;
            finalExplicitRelres = explicitRelres;
            lastResidualNorm = lastRes;
            yg = yNew;

            if flagCPU == 0
                break;
            end

            if explicitRelres <= tol
                vprintf(['  [Outer Top %d] MINRES returned flag=%d but ', ...
                         'true relres=%.3e <= %.3e; accepting current solution.\n'], ...
                        outerIndex,flagCPU,explicitRelres,tol);
                break;
            end

            madeProgress = isfinite(previousExplicitRelres) && ...
                explicitRelres < previousExplicitRelres*(1-minProgressFraction);
            firstNonzeroTermination = ~isfinite(previousExplicitRelres);

            if flagCPU == 3
                % A normal stagnation restart is allowed only when MATLAB
                % actually performed new iterations and the true residual
                % decreased.  iter=0 can NEVER restart the same system.
                if iterCPU > 0 && (firstNonzeroTermination || madeProgress)
                    if isfinite(previousExplicitRelres)
                        if explicitRelres >= previousExplicitRelres*(1-plateauProgressFraction)
                            plateauNoProgressEvents = plateauNoProgressEvents + 1;
                        else
                            plateauNoProgressEvents = 0;
                        end
                        if plateauNoProgressEvents >= plateauMinEvents && ...
                                explicitRelres <= plateauCeiling
                            plateauAccepted = true;
                            plateauAcceptedResidual = explicitRelres;
                            fprintf(['  [Outer Top %d] PLATEAU ACCEPTED after stagnation restarts: ', ...
                                     'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                                    outerIndex,explicitRelres,plateauCeiling,plateauNoProgressEvents);
                            yg = yNew;
                            break;
                        end
                    end
                    stagnationRestarts = stagnationRestarts + 1;
                    recoveryEvents = recoveryEvents + 1;
                    vprintf(['  [Outer Top %d] STAGNATION RESTART %d: ', ...
                             'segment iter=%d, cumulative iter=%d, ', ...
                             'reported relres=%.3e, true relres=%.3e > %.3e.\n'], ...
                            outerIndex,stagnationRestarts,iterCPU,totalIterations, ...
                            relresCPU,explicitRelres,tol);
                    zeroProgressEvents = 0;
                    previousExplicitRelres = explicitRelres;
                    x0 = yNew;
                    continue;
                end

                % Reaching a stagnated main MINRES segment is the
                % TRIGGER for one residual-correction cycle.  Do not count
                % it as a failed recovery yet; otherwise one recovery cycle
                % can be counted twice (once here and once after a failed
                % correction), causing a premature controlled stop.
                if iterCPU == 0
                    vprintf(['  [Outer Top %d] ZERO-ITERATION STAGNATION: ', ...
                             'true relres=%.3e. Switching to residual correction ', ...
                             'instead of restarting the same iterate.\n'], ...
                            outerIndex,explicitRelres);
                else
                    vprintf(['  [Outer Top %d] STAGNATION WITHOUT MATERIAL PROGRESS: ', ...
                             'segment iter=%d, previous relres=%.3e, current relres=%.3e. ', ...
                             'Switching to residual correction.\n'], ...
                            outerIndex,iterCPU,previousExplicitRelres,explicitRelres);
                end

                if residualCorrections >= maxCorrectionCycles
                    if plateauNoProgressEvents >= plateauMinEvents && ...
                            explicitRelres <= plateauCeiling
                        plateauAccepted = true;
                        plateauAcceptedResidual = explicitRelres;
                        fprintf(['  [Outer Top %d] PLATEAU ACCEPTED at correction guard: ', ...
                                 'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                                outerIndex,explicitRelres,plateauCeiling,plateauNoProgressEvents);
                        break;
                    end
                    error(['Residual-correction guard triggered in Outer Top %d after %d ', ...
                           'correction cycles. Current true relres=%.3e. Controlled stop ', ...
                           'prevents an infinite recovery loop.'], ...
                          outerIndex,residualCorrections,explicitRelres);
                end

                [yCorr,corrInfo] = residual_correction(yNew,explicitRelres, ...
                    rhs,rhsNorm,outerIndex,residualCorrections+1);
                residualCorrections = residualCorrections + 1;
                correctionMINRESCalls = correctionMINRESCalls + 1;
                correctionMINRESIterations = correctionMINRESIterations + corrInfo.iterations;
                segmentCount = segmentCount + 1;
                totalIterations = totalIterations + corrInfo.iterations;
                solveSeconds = solveSeconds + corrInfo.solveSeconds;
                residualCheckSeconds = residualCheckSeconds + corrInfo.residualCheckSeconds;
                recoveryEvents = recoveryEvents + 1;


                yg = yCorr;
                finalFlag = corrInfo.flag;
                finalReportedRelres = corrInfo.reportedRelres;
                finalExplicitRelres = corrInfo.afterRelres;
                lastResidualNorm = corrInfo.lastResidualNorm;

                if corrInfo.afterRelres <= tol
                    vprintf(['  [Outer Top %d] RESIDUAL CORRECTION %d CONVERGED: ', ...
                             'corr iter=%d, true relres %.3e -> %.3e.\n'], ...
                            outerIndex,residualCorrections,corrInfo.iterations, ...
                            explicitRelres,corrInfo.afterRelres);
                    break;
                end

                % Count plateau behavior once per COMPLETE correction cycle.
                % A reduction smaller than plateauProgressFraction is treated
                % as remaining on the same numerical residual platform.
                if corrInfo.afterRelres >= explicitRelres*(1-plateauProgressFraction)
                    plateauNoProgressEvents = plateauNoProgressEvents + 1;
                else
                    plateauNoProgressEvents = 0;
                end
                if plateauNoProgressEvents >= plateauMinEvents && ...
                        corrInfo.afterRelres <= plateauCeiling
                    plateauAccepted = true;
                    plateauAcceptedResidual = corrInfo.afterRelres;
                    fprintf(['  [Outer Top %d] PLATEAU ACCEPTED after residual corrections: ', ...
                             'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                            outerIndex,corrInfo.afterRelres,plateauCeiling,plateauNoProgressEvents);
                    break;
                end

                if corrInfo.afterRelres < explicitRelres*(1-minProgressFraction)
                    vprintf(['  [Outer Top %d] RESIDUAL CORRECTION %d IMPROVED: ', ...
                             'corr iter=%d, true relres %.3e -> %.3e; ', ...
                             'returning to main MINRES from corrected iterate.\n'], ...
                            outerIndex,residualCorrections,corrInfo.iterations, ...
                            explicitRelres,corrInfo.afterRelres);
                    zeroProgressEvents = 0;
                    previousExplicitRelres = corrInfo.afterRelres;
                    x0 = yCorr;
                    continue;
                end

                % Count one event only after a COMPLETE correction
                % cycle has failed to make material progress.
                zeroProgressEvents = zeroProgressEvents + 1;
                vprintf(['  [Outer Top %d] RESIDUAL CORRECTION %d MADE NO MATERIAL PROGRESS: ', ...
                         'true relres %.3e -> %.3e (failed recovery cycle %d/%d).\n'], ...
                        outerIndex,residualCorrections,explicitRelres,corrInfo.afterRelres, ...
                        zeroProgressEvents,maxZeroProgressEvents);

                if zeroProgressEvents >= maxZeroProgressEvents
                    if plateauNoProgressEvents >= plateauMinEvents && ...
                            corrInfo.afterRelres <= plateauCeiling
                        plateauAccepted = true;
                        plateauAcceptedResidual = corrInfo.afterRelres;
                        fprintf(['  [Outer Top %d] PLATEAU ACCEPTED at zero-progress guard: ', ...
                                 'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                                outerIndex,corrInfo.afterRelres,plateauCeiling,plateauNoProgressEvents);
                        break;
                    end
                    error(['Residual correction failed to reduce the residual in Outer Top %d. ', ...
                           'After %d failed recovery cycles, true relres=%.3e. Controlled stop ', ...
                           'prevents an infinite loop.'], ...
                          outerIndex,zeroProgressEvents,corrInfo.afterRelres);
                end
                previousExplicitRelres = corrInfo.afterRelres;
                x0 = yCorr;
                continue;
            end

            if flagCPU == 1
                if iterCPU <= 0
                    error(['MINRES returned flag=1 with zero iterations in Outer Top %d. ', ...
                           'Stopping because the state cannot advance safely.'],outerIndex);
                end
                if isfinite(previousExplicitRelres)
                    if explicitRelres >= previousExplicitRelres*(1-plateauProgressFraction)
                        plateauNoProgressEvents = plateauNoProgressEvents + 1;
                    else
                        plateauNoProgressEvents = 0;
                    end
                    if plateauNoProgressEvents >= plateauMinEvents && ...
                            explicitRelres <= plateauCeiling
                        plateauAccepted = true;
                        plateauAcceptedResidual = explicitRelres;
                        fprintf(['  [Outer Top %d] PLATEAU ACCEPTED after MAXIT continuations: ', ...
                                 'true relres=%.3e <= %.3e after %d plateau events.\n'], ...
                                outerIndex,explicitRelres,plateauCeiling,plateauNoProgressEvents);
                        yg = yNew;
                        break;
                    end
                end
                maxitRestarts = maxitRestarts + 1;
                recoveryEvents = recoveryEvents + 1;
                vprintf(['  [Outer Top %d] MAXIT CONTINUATION %d: ', ...
                         'segment maxit=%d reached, cumulative iter=%d, ', ...
                         'true relres=%.3e > %.3e; continuing from current iterate.\n'], ...
                        outerIndex,maxitRestarts,segmentMaxit,totalIterations, ...
                        explicitRelres,tol);
                previousExplicitRelres = explicitRelres;
                x0 = yNew;
                continue;
            end

            error(['Preconditioned MINRES encountered a genuine numerical ', ...
                   'breakdown in Outer Top %d: flag=%d, reported relres=%.3e, ', ...
                   'true relres=%.3e, cumulative iter=%d.'], ...
                  outerIndex,flagCPU,relresCPU,explicitRelres,totalIterations);
        end

        info = struct();
        info.totalIterations = totalIterations;
        info.segmentCount = segmentCount;
        info.stagnationRestarts = stagnationRestarts;
        info.maxitRestarts = maxitRestarts;
        info.residualCorrections = residualCorrections;
        info.correctionMINRESCalls = correctionMINRESCalls;
        info.correctionMINRESIterations = correctionMINRESIterations;
        info.zeroProgressEvents = zeroProgressEvents;
        info.recoveryEvents = recoveryEvents;
        info.plateauAccepted = plateauAccepted;
        info.plateauAcceptedResidual = plateauAcceptedResidual;
        info.plateauNoProgressEvents = plateauNoProgressEvents;
        info.solveSeconds = solveSeconds;
        info.residualCheckSeconds = residualCheckSeconds;
        info.finalFlag = finalFlag;
        info.finalReportedRelres = finalReportedRelres;
        info.finalExplicitRelres = finalExplicitRelres;
        info.lastResidualNorm = lastResidualNorm;

        function [yCorr,cinfo] = residual_correction(yBase,beforeRelres, ...
                rhsLocal,rhsNormLocal,outerIdx,correctionIndex)
            % Solve A*delta = r with r = rhs-A*yBase.  The correction solve
            % uses a RELATIVE tolerance chosen so that its residual should be
            % comfortably below the original absolute target tol*||rhs||.
            % This avoids asking the tiny correction equation for another
            % 1e-10 relative solve, which is unnecessary and numerically
            % counterproductive.

            checkTimer = tic;
            rCorr = rhsLocal - Ashift(yBase);
            wait_for_gpu(rCorr);
            rCorrNorm = double(gather_scalar(norm(rCorr,2)));
            beforeRelresMeasured = rCorrNorm/rhsNormLocal;
            checkSeconds = toc(checkTimer);

            if ~isfinite(beforeRelresMeasured)
                error('Residual correction received nonfinite residual in Outer Top %d.',outerIdx);
            end
            if beforeRelresMeasured <= tol
                yCorr = yBase;
                cinfo = struct('iterations',0,'solveSeconds',0, ...
                    'residualCheckSeconds',checkSeconds,'flag',0, ...
                    'reportedRelres',beforeRelresMeasured, ...
                    'afterRelres',beforeRelresMeasured,'lastResidualNorm',rCorrNorm, ...
                    'correctionTolerance',NaN);
                return;
            end

            corrTol = correctionSafety*tol/max(beforeRelresMeasured,realmin('double'));
            corrTol = min(correctionMaxTol,max(correctionMinTol,corrTol));
            delta0 = zeros(size(rCorr),'like',rCorr);

            warnState2 = warning;
            warnCleanup2 = onCleanup(@()warning(warnState2)); %#ok<NASGU>
            warning('off','all');

            corrMaxit = maxit;
            corrTimer = tic;
            [delta,cflag,crelres,citer,cresvec] = minres( ...
                Ashift,rCorr,corrTol,corrMaxit,Pinv,[],delta0);
            wait_for_gpu(delta);
            corrSeconds = toc(corrTimer);

            warning(warnState2);
            clear warnCleanup2

            cflagCPU = double(gather_scalar(cflag));
            crelresCPU = double(gather_scalar(crelres));
            citerCPU = double(gather_scalar(citer));
            if isempty(cresvec)
                cLastRes = NaN;
            else
                cLastRes = double(gather_scalar(cresvec(end)));
            end
            if any(~isfinite([cflagCPU,crelresCPU,citerCPU]))
                error('Correction MINRES produced nonfinite statistics in Outer Top %d.',outerIdx);
            end

            yCorr = yBase + delta;
            [afterRelres,~,postCheckSeconds] = explicit_relative_residual( ...
                yCorr,rhsLocal,rhsNormLocal);
            checkSeconds = checkSeconds + postCheckSeconds;

            vprintf(['  [Outer Top %d] RESIDUAL CORRECTION %d: ', ...
                     'before=%.3e (measured %.3e), corr tol=%.3e, ', ...
                     'corr iter=%d, corr flag=%d, corr reported relres=%.3e, ', ...
                     'after=%.3e.\n'], ...
                    outerIdx,correctionIndex,beforeRelres,beforeRelresMeasured, ...
                    corrTol,citerCPU,cflagCPU,crelresCPU,afterRelres);

            cinfo = struct();
            cinfo.iterations = citerCPU;
            cinfo.solveSeconds = corrSeconds;
            cinfo.residualCheckSeconds = checkSeconds;
            cinfo.flag = cflagCPU;
            cinfo.reportedRelres = crelresCPU;
            cinfo.afterRelres = afterRelres;
            cinfo.lastResidualNorm = cLastRes;
            cinfo.correctionTolerance = corrTol;
        end

        function [relresTrue,rNorm,seconds] = explicit_relative_residual(yTest,rhsLocal,rhsNormLocal)
            tcheck = tic;
            rTest = rhsLocal - Ashift(yTest);
            wait_for_gpu(rTest);
            rNorm = double(gather_scalar(norm(rTest,2)));
            relresTrue = rNorm/rhsNormLocal;
            seconds = toc(tcheck);
            clear rTest
            if ~isfinite(relresTrue)
                error('Explicit residual became nonfinite in Outer Top %d.',outerIndex);
            end
        end
    end

    function vprintf(varargin)
        if verbose
            fprintf(varargin{:});
        end
    end

    function s = getStatsLocal()
        s = stats;
        if s.totalMINRESSolves > 0
            s.averageMINRESIterations = s.totalMINRESIterations/s.totalMINRESSolves;
            s.averageMINRESTime = s.totalMINRESTime/s.totalMINRESSolves;
            s.averageMINRESCallsPerTop = s.totalMINRESCalls/s.totalMINRESSolves;
        else
            s.averageMINRESIterations = NaN;
            s.averageMINRESTime = NaN;
            s.averageMINRESCallsPerTop = NaN;
        end
        if s.topFunctionCalls > 0
            s.averageTopTime = s.totalTopTime/s.topFunctionCalls;
            s.averageH2DTime = s.totalH2DTime/s.topFunctionCalls;
            s.averageD2HTime = s.totalD2HTime/s.topFunctionCalls;
        else
            s.averageTopTime = NaN;
            s.averageH2DTime = NaN;
            s.averageD2HTime = NaN;
        end
        s.unaccountedTopTime = max( ...
            s.totalTopTime - s.totalH2DTime - s.totalD2HTime ...
            - s.totalRHSMapTime - s.totalMINRESTime - s.totalResidualCheckTime,0);
    end
end

function s = init_stats()
    s.topFunctionCalls = 0;
    s.topVectorApplications = 0;
    s.unexpectedGPUInputCalls = 0;

    s.totalMINRESSolves = 0;
    s.totalMINRESCalls = 0;
    s.totalMINRESIterations = 0;
    s.totalMINRESTime = 0;
    s.totalResidualCheckTime = 0;
    s.totalRHSMapTime = 0;
    s.totalH2DTime = 0;
    s.totalD2HTime = 0;
    s.totalTopTime = 0;
    s.totalStagnationRestarts = 0;
    s.totalMaxitRestarts = 0;
    s.totalResidualCorrections = 0;
    s.totalCorrectionMINRESCalls = 0;
    s.totalCorrectionMINRESIterations = 0;
    s.totalZeroProgressEvents = 0;
    s.totalRecoveryEvents = 0;
    s.totalPlateauAccepts = 0;
    s.maxPlateauAcceptedResidual = 0;
    s.plateauAcceptedResiduals = zeros(0,1);

    s.maxMINRESIterations = 0;
    s.maxMINRESRelativeResidual = 0;

    s.flags = zeros(0,1);
    s.iterations = zeros(0,1);
    s.relativeResiduals = zeros(0,1);
    s.explicitRelativeResiduals = zeros(0,1);
    s.lastResidualNorms = zeros(0,1);
    s.solveTimes = zeros(0,1);
    s.residualCheckTimes = zeros(0,1);
    s.rhsTimes = zeros(0,1);
    s.segmentCounts = zeros(0,1);
    s.stagnationRestartsPerTop = zeros(0,1);
    s.maxitRestartsPerTop = zeros(0,1);
    s.residualCorrectionsPerTop = zeros(0,1);
    s.correctionMINRESIterationsPerTop = zeros(0,1);
    s.zeroProgressEventsPerTop = zeros(0,1);
    s.recoveryEventsPerTop = zeros(0,1);
    s.plateauAcceptedPerTop = zeros(0,1);
    s.plateauAcceptedResidualPerTop = zeros(0,1);
    s.h2dTimes = zeros(0,1);
    s.d2hTimes = zeros(0,1);
    s.topTimes = zeros(0,1);
end
