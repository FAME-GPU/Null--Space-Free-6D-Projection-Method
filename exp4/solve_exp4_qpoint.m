function qres = solve_exp4_qpoint(st,cfg,qCPU,pointIndex,selectedMode)
%SOLVE_EXP4_QPOINT Solve one Bloch point with all large calculations on GPU.
    if nargin<5, selectedMode=0; end
    sys=build_exp4_q_system(st,qCPU,cfg);

    % Generate the large starting vector directly on the GPU.
    seed=cfg.solver.rngSeed + 10007*pointIndex;
    rng(seed,'twister');
    parallel.gpu.rng(seed,'Philox4x32-10');
    v0=randn(sys.nr,1,'like',st.P)+1i*randn(sys.nr,1,'like',st.P);

    [Aop,getStats]=make_krinv_instrumented( ...
        sys,cfg.solver.tolCG,cfg.solver.maxitInnerCG);

    if selectedMode>0, sel=selectedMode; else, sel=[]; end
    wait_for_gpu(); t=tic;
    [Ysel,theta,outerSteps,info,selIdx]=Lanczos_spectrum_selected( ...
        @(x)Aop(x),@(x)x,@(x)x,sys.nr,cfg.spectrum.nev, ...
        cfg.solver.tolLanczos,cfg.solver.maxLanczosRestarts, ...
        cfg.spectrum.p,v0,true,cfg.solver.verboseLanczos,sel);
    wait_for_gpu(Ysel); wall=toc(t);

    lambdaGPU=real(1./theta(:));
    lambda=gather_exp4_for_io(lambdaGPU);
    cgStats=getStats();

    if cfg.spectrum.requireFullConvergence && (~info.converged || ...
            any(~isfinite(info.allRelativeInverseRitzResiduals)) || ...
            max(info.allRelativeInverseRitzResiduals)>cfg.solver.tolLanczos)
        error(['Exp4 point %d did not converge all %d wanted eigenvalues: ', ...
            'numConverged=%d, restart=%d/%d.'], ...
            pointIndex,cfg.spectrum.nev,info.numConverged,info.finalRestart, ...
            cfg.solver.maxLanczosRestarts);
    end

    Xselected=[]; recoveryInfo=[];
    if selectedMode>0
        if isempty(selIdx) || selIdx(1)~=selectedMode
            error('Selected mode extraction mismatch at point %d.',pointIndex);
        end
        [XselectedGPU,recoveryInfo]=recover_fixed_vectors_tracking(Ysel,sys,cfg);
        % This gather is serialization only: the checkpoint must survive GPU release.
        Xselected=gather_exp4_for_io(XselectedGPU);
        clear XselectedGPU
    end

    qres=struct();
    qres.pointIndex=pointIndex;
    qres.q=qCPU(:);
    qres.lambda=lambda;
    qres.outerSteps=outerSteps;
    qres.wallTimeSeconds=wall;
    qres.converged=info.converged;
    qres.numConverged=info.numConverged;
    qres.finalRestart=info.finalRestart;
    qres.maxRelativeInverseRitzResidual=max(info.allRelativeInverseRitzResiduals);
    qres.cgAverageIterations=cgStats.averageIterations;
    qres.cgMaxIterations=cgStats.maxIterations;
    qres.cgMaxRelativeResidual=cgStats.maxRelativeResidual;
    qres.selectedMode=selectedMode;
    qres.selectedLambda=[];
    qres.Xselected=Xselected;
    qres.recoveryInfo=recoveryInfo;
    if selectedMode>0, qres.selectedLambda=lambda(selectedMode); end

    clear Ysel theta lambdaGPU v0 sys Aop
end
