function out = solve_tracking_scan(sys,cfg,kScan,pScan,fixedIndices,targetLambda)
%SOLVE_TRACKING_SCAN Compute a low-spectrum scan and selected Ritz vectors.

    rng(cfg.solver.rngSeed + 1000*sys.N + kScan,'twister');
    v0cpu=randn(sys.nr,1)+1i*randn(sys.nr,1);
    if isa(sys.P,'gpuArray')
        v0=gpuArray(cast(v0cpu,classUnderlying(sys.P)));
    else
        v0=cast(v0cpu,'like',sys.P);
    end
    clear v0cpu

    [Aop,getStats]=make_krinv_instrumented( ...
        sys,cfg.solver.tolCG,cfg.solver.maxitInnerCG);

    if isempty(fixedIndices)
        halfWidth=cfg.scan.N5.candidateHalfWidth;
    else
        halfWidth=0;
    end

    wait_for_gpu();
    t=tic;
    [Ysel,theta,outerSteps,info,selIdx]=Lanczos_tracking( ...
        @(x)Aop(x),@(x)x,@(x)x,sys.nr,kScan,cfg.solver.tolLanczos, ...
        cfg.solver.maxLanczosRestarts,pScan,v0,isa(v0,'gpuArray'), ...
        cfg.solver.verboseLanczos,fixedIndices,targetLambda,halfWidth);
    wait_for_gpu(Ysel);
    wall=toc(t);

    lambda=real(1./theta(:));
    stats=getStats();

    out=struct();
    out.kScan=kScan; out.pScan=pScan;
    out.lambda=lambda;
    out.theta=theta(:);
    out.Yselected=Ysel;
    out.selectedIndices=selIdx(:);
    out.selectedLambda=lambda(selIdx);
    out.selectedRelRitzResidual=info.selectedRelativeInverseRitzResiduals(:);
    out.outerSteps=outerSteps;
    out.info=info;
    out.cgStats=stats;
    out.wallTimeSeconds=wall;

    if cfg.output.verbose
        fprintf(['  scan N=%d: k=%d, p=%d, outer=%d, wall=%.1f s, ', ...
            'Mhat-CG avg=%.2f, lambda range=[%.6e, %.6e]\n'], ...
            sys.N,kScan,pScan,outerSteps,wall,stats.averageIterations, ...
            lambda(1),lambda(end));
        if info.hitMaxRestart
            fprintf('    note: scan hit restart budget; selected Ritz residuals are checked separately.\n');
        end
    end
end
