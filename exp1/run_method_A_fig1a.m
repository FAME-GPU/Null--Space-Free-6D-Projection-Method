function one = run_method_A_fig1a(sys,cfg,seed,epsMean)
%RUN_METHOD_A_FIG1A Run frozen Method A for one Bloch point.
%
% The eigensolve timing ends before the independent original-GEVP residual
% diagnostics.  All MINRES work, including restarts and residual-correction
% MINRES calls, is included in totalLinearSolverIterations.

    wait(gpuDevice);
    setupTimer = tic;
    [Top,getStats,meta] = fig1a_make_A_eigs_top_preconditioned(sys,cfg,epsMean);
    wait(gpuDevice);
    preconditionerBuildSeconds = toc(setupTimer);

    rng(seed,'twister');
    v0 = complex(randn(sys.ndof,1),randn(sys.ndof,1));
    v0 = v0/norm(v0,2);

    s0 = getStats();
    top0 = s0.topVectorApplications;
    calls0 = s0.topFunctionCalls;
    it0 = s0.totalMINRESIterations;

    wait(gpuDevice);
    eigTimer = tic;
    [U,Theta,eigsFlag] = eigs( ...
        Top,sys.ndof,cfg.spectrum.numEig,'largestreal', ...
        'Tolerance',cfg.eigs.tolerance, ...
        'MaxIterations',cfg.eigs.maxIterations, ...
        'SubspaceDimension',cfg.eigs.subspaceDimension, ...
        'StartVector',v0, ...
        'Display',cfg.eigs.display, ...
        'FailureTreatment','keep', ...
        'IsFunctionSymmetric',false);
    wait(gpuDevice);
    eigsWallSeconds = toc(eigTimer);

    if isa(U,'gpuArray') || isa(Theta,'gpuArray')
        error('A-only Fig. 1(a) requires CPU eigs Krylov vectors.');
    end

    s = getStats();
    outer = s.topVectorApplications-top0;
    topCalls = s.topFunctionCalls-calls0;
    lin = s.totalMINRESIterations-it0;

    theta = diag(Theta);
    lambdaRaw = cfg.shift.sigma + 1./theta;
    [lambda,idx] = sort(real(lambdaRaw),'ascend');
    U = U(:,idx);

    % R2024b/Linux function-handle eigs has been observed to return a
    % non-scalar convergence-status array.  Always reduce it explicitly.
    eigsFlagRaw = double(eigsFlag);
    eigsConverged = ~isempty(eigsFlagRaw) && ...
        all(isfinite(eigsFlagRaw(:))) && all(eigsFlagRaw(:)==0);

    converged = eigsConverged && ...
        numel(lambda)==cfg.spectrum.numEig && ...
        all(isfinite(lambda)) && all(lambda>0);

    residualTimer = tic;
    gevpResiduals = NaN(numel(lambda),1);
    nullMDefects = NaN(numel(lambda),1);
    if cfg.diagnostics.computeOriginalGEVPResidual && converged
        for j = 1:numel(lambda)
            ug = to_gpu(U(:,j),cfg.gpu.precision);
            Ku = sys.Kop(ug);
            Mu = sys.mass.Mop(ug);
            r = Ku-lambda(j)*Mu;
            wait_for_gpu(r);

            den = gather_scalar(norm(Ku,2)) + ...
                  abs(lambda(j))*gather_scalar(norm(Mu,2));
            gevpResiduals(j) = gather_scalar(norm(r,2))/max(den,realmin);
            nullMDefects(j) = gather_scalar(norm(sys.U0Hop(Mu),2))/ ...
                max(gather_scalar(norm(Mu,2)),realmin);
            clear ug Ku Mu r
        end
    end
    residualDiagnosticSeconds = toc(residualTimer);

    maxGEVPResidual = max(gevpResiduals,[],'omitnan');
    maxNullMDefect = max(nullMDefects,[],'omitnan');
    if isempty(maxGEVPResidual), maxGEVPResidual = NaN; end
    if isempty(maxNullMDefect), maxNullMDefect = NaN; end

    one = struct();
    one.lambda = lambda(:);
    one.theta = theta(idx);
    one.outerSteps = outer;
    one.topFunctionCalls = topCalls;
    one.totalLinearSolverIterations = lin;
    one.eigsFlag = double(~eigsConverged);
    one.eigsFlagRaw = eigsFlagRaw(:);
    one.converged = converged;

    one.eigensolveSeconds = eigsWallSeconds;
    one.preconditionerBuildSeconds = preconditionerBuildSeconds;
    one.residualDiagnosticSeconds = residualDiagnosticSeconds;

    one.totalMINRESSolves = s.totalMINRESSolves;
    one.totalMINRESCalls = s.totalMINRESCalls;
    one.totalMINRESIterations = s.totalMINRESIterations;
    one.averageMINRESIterations = s.averageMINRESIterations;
    one.maxMINRESIterations = s.maxMINRESIterations;
    one.maxInnerFinalRelativeResidual = s.maxMINRESRelativeResidual;
    one.totalStagnationRestarts = s.totalStagnationRestarts;
    one.totalMaxitRestarts = s.totalMaxitRestarts;
    one.totalResidualCorrections = s.totalResidualCorrections;
    one.totalCorrectionMINRESCalls = s.totalCorrectionMINRESCalls;
    one.totalCorrectionMINRESIterations = s.totalCorrectionMINRESIterations;
    one.totalRecoveryEvents = s.totalRecoveryEvents;

    one.totalPlateauAccepts = s.totalPlateauAccepts;
    if s.totalPlateauAccepts > 0
        one.maxPlateauAcceptedResidual = s.maxPlateauAcceptedResidual;
    else
        one.maxPlateauAcceptedResidual = NaN;
    end
    one.plateauAcceptedResiduals = s.plateauAcceptedResiduals;
    one.plateauAcceptedPerTop = s.plateauAcceptedPerTop;
    one.plateauAcceptedResidualPerTop = s.plateauAcceptedResidualPerTop;

    one.minresIterationsPerTop = s.iterations;
    one.innerFinalRelativeResiduals = s.explicitRelativeResiduals;
    one.stagnationRestartsPerTop = s.stagnationRestartsPerTop;
    one.residualCorrectionsPerTop = s.residualCorrectionsPerTop;
    one.correctionMINRESIterationsPerTop = s.correctionMINRESIterationsPerTop;

    one.gevpResiduals = gevpResiduals;
    one.nullMDefects = nullMDefects;
    one.maxOriginalGEVPResidual = maxGEVPResidual;
    one.maxNullMDefect = maxNullMDefect;
    one.preconditioner = meta.preconditioner;

    if converged && isfinite(maxGEVPResidual) && ...
            maxGEVPResidual > cfg.diagnostics.gevpResidualWarning
        warning(['A-only Fig. 1(a): final original-GEVP residual %.3e exceeds ', ...
                 'the diagnostic warning level %.3e. Result is saved for inspection.'], ...
                maxGEVPResidual,cfg.diagnostics.gevpResidualWarning);
    end

    clear U v0
end
