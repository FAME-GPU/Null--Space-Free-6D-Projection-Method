function one = run_method_A_fig1bc(sys,cfg,seed,epsMean,budget)
%RUN_METHOD_A_FIG1BC Method A for Fig. 1(b,c), with recorded plateau fallback.

    budget.checkCanContinue('Method A preconditioner setup');
    wait(gpuDevice); setupTimer=tic;
    [Top,getStats,meta]=make_A_eigs_top_preconditioned(sys,cfg,epsMean,budget);
    wait(gpuDevice); preconditionerBuildSeconds=toc(setupTimer);
    budget.checkCanContinue('Method A eigs start');

    rng(seed,'twister');
    v0=complex(randn(sys.ndof,1),randn(sys.ndof,1));
    v0=v0/norm(v0,2);
    s0=getStats(); top0=s0.topVectorApplications; calls0=s0.topFunctionCalls; it0=s0.totalMINRESIterations;

    wait(gpuDevice); eigTimer=tic;
    [U,Theta,eigsFlag]=eigs(Top,sys.ndof,cfg.spectrum.numEig,'largestreal', ...
        'Tolerance',cfg.eigs.tolerance,'MaxIterations',cfg.eigs.maxIterations, ...
        'SubspaceDimension',cfg.eigs.subspaceDimension,'StartVector',v0, ...
        'Display',cfg.eigs.display,'FailureTreatment','keep','IsFunctionSymmetric',false);
    wait(gpuDevice); eigsWallSeconds=toc(eigTimer);
    budget.check('Method A eigs return');

    if isa(U,'gpuArray') || isa(Theta,'gpuArray')
        error('EXP1:UnexpectedGPUeigs','Method A requires CPU eigs Krylov vectors.');
    end
    s=getStats();
    outer=s.topVectorApplications-top0; topCalls=s.topFunctionCalls-calls0;
    lin=s.totalMINRESIterations-it0;
    theta=diag(Theta); lambdaRaw=cfg.shift.sigma+1./theta;
    [lambda,idx]=sort(real(lambdaRaw),'ascend'); %#ok<ASGLU>

    eigsFlagRaw=double(eigsFlag);
    eigsConverged=~isempty(eigsFlagRaw) && all(isfinite(eigsFlagRaw(:))) && all(eigsFlagRaw(:)==0);
    converged=eigsConverged && numel(lambda)==cfg.spectrum.numEig && ...
        all(isfinite(lambda)) && all(lambda>0);

    one=struct();
    one.lambda=lambda(:); one.outerSteps=outer; one.inverseCalls=outer;
    one.topFunctionCalls=topCalls; one.totalLinearSolverIterations=lin;
    one.primaryLinearIterations=lin; one.nestedLinearIterations=0;
    one.eigsFlag=double(~eigsConverged); one.eigsFlagRaw=eigsFlagRaw(:);
    one.converged=converged;
    one.eigensolveSeconds=eigsWallSeconds; one.preconditionerBuildSeconds=preconditionerBuildSeconds;
    one.dominantLinearSolveSeconds=s.totalMINRESTime;
    one.remainingInverseSeconds=max(s.totalTopTime-s.totalMINRESTime,0);
    one.outerLanczosSeconds=max(eigsWallSeconds-s.totalTopTime,0);
    one.totalWallSeconds=eigsWallSeconds;
    one.maxFinalRitzResidual=NaN; one.maxLinearRelativeResidual=s.maxMINRESRelativeResidual;
    one.averagePrimaryLinearIterations=s.averageMINRESIterations;
    one.maxPrimaryLinearIterations=s.maxMINRESIterations;
    one.averageNestedLinearIterations=0; one.maxNestedLinearIterations=0;
    one.totalMINRESSolves=s.totalMINRESSolves;
    one.totalMINRESCalls=s.totalMINRESCalls;
    one.totalStagnationRestarts=s.totalStagnationRestarts;
    one.totalMaxitRestarts=s.totalMaxitRestarts;
    one.totalResidualCorrections=s.totalResidualCorrections;
    one.totalCorrectionMINRESCalls=s.totalCorrectionMINRESCalls;
    one.totalCorrectionMINRESIterations=s.totalCorrectionMINRESIterations;
    one.totalRecoveryEvents=s.totalRecoveryEvents;
    one.totalPlateauAccepts=s.totalPlateauAccepts;
    if s.totalPlateauAccepts>0, one.maxPlateauAcceptedResidual=s.maxPlateauAcceptedResidual; else, one.maxPlateauAcceptedResidual=NaN; end
    one.plateauAcceptedResiduals=s.plateauAcceptedResiduals;
    one.plateauAcceptedPerTop=s.plateauAcceptedPerTop;
    one.plateauAcceptedResidualPerTop=s.plateauAcceptedResidualPerTop;
    one.innerFinalRelativeResiduals=s.explicitRelativeResiduals;
    one.minresIterationsPerTop=s.iterations;
    one.preconditioner=meta.preconditioner;
    clear U v0
end
