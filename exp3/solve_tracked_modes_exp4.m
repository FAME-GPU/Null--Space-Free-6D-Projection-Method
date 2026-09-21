function selected = solve_tracked_modes_exp4(sys,cfg)
%SOLVE_TRACKED_MODES_EXP4 Solve N=5 scan and recover only modes 11 and 114.

    modes=cfg.tracking.n5Modes(:).';
    kScan=cfg.scan.N5.k;
    pScan=kScan+cfg.scan.N5.pExtra;
    if max(modes)>kScan
        error('kScan=%d does not include tracked mode %d.',kScan,max(modes));
    end

    % Build the projected physical wave vectors before the expensive
    % Lanczos scan.  This is inexpensive and makes any geometry/interface
    % error fail immediately rather than after the N=5 eigensolve.
    qModes=build_projected_q_modes( ...
        sys.P,sys.kLift,sys.fourier.Xi,sys.NF,cfg.problem.dim);
    qModesCPU=gather(qModes);
    clear qModes

    scan=solve_tracking_scan(sys,cfg,kScan,pScan,modes,[]);

    % The production run must not silently accept an unconverged scan.
    if ~scan.info.converged
        error(['N=5 selected-mode Lanczos did not satisfy the requested ', ...
            'convergence test (final restart %d, hitMaxRestart=%d).'], ...
            scan.info.finalRestart,scan.info.hitMaxRestart);
    end
    if ~isequal(scan.selectedIndices(:).',modes)
        error('Selected Ritz indices do not match requested tracked modes.');
    end
    if any(scan.selectedRelRitzResidual>cfg.solver.requiredSelectedRelRitzResidual)
        error('Selected relative inverse-Ritz residual exceeds required threshold.');
    end

    [Xcpu,recoveryInfo]=recover_fixed_vectors_tracking(scan.Yselected,sys,cfg);
    lambda=scan.selectedLambda(:).';

    relCheck=abs(lambda-cfg.tracking.n5LambdaReference) ./ ...
        max(abs(cfg.tracking.n5LambdaReference),eps);
    if any(relCheck>cfg.tracking.lambdaCheckRelTol)
        error(['Tracked-mode eigenvalue sanity check failed.  Computed relative ', ...
            'differences are [%g, %g].'],relCheck(1),relCheck(2));
    end

    selected=struct();
    selected.N=cfg.problem.N;
    selected.NF=sys.NF;
    selected.modes=modes;
    selected.lambda6D=lambda;
    selected.Xcpu=Xcpu;
    selected.qModesCPU=qModesCPU;
    selected.outerSteps=scan.outerSteps;
    selected.scanWallTimeSeconds=scan.wallTimeSeconds;
    selected.MhatCGAverageIterations=scan.cgStats.averageIterations;
    selected.selectedRelativeInverseRitzResidual=scan.selectedRelRitzResidual(:).';
    selected.recoveryIterations=recoveryInfo.iterations(:).';
    selected.recoveryRelativeResiduals=recoveryInfo.relativeResiduals(:).';
    selected.referenceTracking=cfg.tracking;
end
