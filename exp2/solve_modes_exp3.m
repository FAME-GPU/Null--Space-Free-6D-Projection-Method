function sol = solve_modes_exp3(sys,cfg,tolCG,tolLan,seed)
%SOLVE_MODES_EXP3 Compute the first positive modes with timed sub-stages.
%
% The eigensolver breakdown is stored for possible later use:
%   - inner Mhat-CG time inside explicit K_r^{-1};
%   - remaining explicit inverse-action time;
%   - outer Lanczos algebra time;
%   - eigenvector recovery time;
%   - Fourier-diagnostic time.
%
% Only the aggregate eigensolver time is used in Figure 3(b).

    rng(seed,'twister');
    v0cpu=randn(sys.nr,1)+1i*randn(sys.nr,1);
    if isa(sys.P,'gpuArray')
        v0=gpuArray(cast(v0cpu,classUnderlying(sys.P)));
    else
        v0=cast(v0cpu,'like',sys.P);
    end

    [KrInvOp,getCGStats]=make_krinv_instrumented( ...
        sys,tolCG,cfg.solver.maxitInnerCG);

    fncA = @(x) KrInvOp(x);
    fncB = @(x) x;
    fncBinv = @(x) x;
    flagGPU = isa(v0,'gpuArray');

    wait_for_gpu();
    tLanczos=tic;
    [Y,theta,m,lanczosInfo]=Lanczos( ...
        fncA,fncB,fncBinv,sys.nr,cfg.spectrum.numEig,tolLan, ...
        cfg.solver.maxLanczosRestarts,cfg.solver.krylovDim,v0,flagGPU, ...
        cfg.solver.verboseLanczos);
    wait_for_gpu(Y);
    lanczosCoreSeconds=toc(tLanczos);

    lambda=real(1./theta(:));
    [lambda,idx]=sort(lambda,'ascend');
    Y=Y(:,idx);

    wait_for_gpu();
    tRecovery=tic;
    [X,MX,recoveryInfo]=recover_original_eigenvectors_exp2(Y,sys,cfg);
    wait_for_gpu(X);
    recoverySeconds=toc(tRecovery);

    wait_for_gpu();
    tDiag=tic;
    diagOut=compute_fourier_diagnostics_exp2(X,MX,lambda,sys);
    wait_for_gpu();
    diagnosticSeconds=toc(tDiag);

    cgStats=getCGStats();
    innerCGSeconds=cgStats.totalInnerCGSeconds;
    inverseActionOtherSeconds=cgStats.totalInverseActionOtherSeconds;
    outerLanczosAlgebraSeconds=max(0,lanczosCoreSeconds-cgStats.totalActionSeconds);
    eigensolverTotalSeconds=lanczosCoreSeconds+recoverySeconds+diagnosticSeconds;

    sol=struct();
    sol.lambda=lambda;
    sol.X=X;
    sol.MX=MX;
    sol.Y=Y;
    sol.resGEVP=diagOut.resGEVP;
    sol.divRelative2=diagOut.divRelative2;
    sol.rayleigh=diagOut.rayleigh;
    sol.outerSteps=m;
    sol.cgStats=cgStats;
    sol.recoveryInfo=recoveryInfo;
    sol.lanczosInfo=lanczosInfo;
    sol.wallTimeSeconds=eigensolverTotalSeconds;
    sol.timing=struct( ...
        'totalEigensolverSeconds',eigensolverTotalSeconds, ...
        'lanczosCoreSeconds',lanczosCoreSeconds, ...
        'innerMhatCGSeconds',innerCGSeconds, ...
        'inverseActionOtherSeconds',inverseActionOtherSeconds, ...
        'outerLanczosAlgebraSeconds',outerLanczosAlgebraSeconds, ...
        'eigenvectorRecoverySeconds',recoverySeconds, ...
        'fourierDiagnosticsSeconds',diagnosticSeconds);

    if cfg.output.verbose
        fprintf(['  inverse Lanczos: outer=%d, inner Mhat CG=%d, ', ...
            'max r_GEVP=%.3e, eigensolver wall=%.2f s\n'], ...
            m,cgStats.totalIterations,max(sol.resGEVP),eigensolverTotalSeconds);
        fprintf(['    timing: Mhat CG %.2f s, inverse-other %.2f s, ', ...
            'outer algebra %.2f s, recovery %.2f s, diagnostics %.2f s\n'], ...
            innerCGSeconds,inverseActionOtherSeconds,outerLanczosAlgebraSeconds, ...
            recoverySeconds,diagnosticSeconds);
    end
end
