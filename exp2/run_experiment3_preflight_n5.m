function run_experiment3_preflight_n5(cfg,rootDir)
%RUN_EXPERIMENT3_PREFLIGHT_N5 Production-size backend/action validation.

    if nargin<2, rootDir=fileparts(mfilename('fullpath')); end %#ok<NASGU>
    gpuInfo=gpu_initialize(cfg.gpu);

    fprintf('[Exp3 N=5 preflight] building production system...\n');
    wait_for_gpu(); t0=tic;
    sys=build_experiment3_system(cfg,5,gpuInfo);
    wait_for_gpu(); buildSeconds=toc(t0);
    fprintf('  build %.2f s, M=%d, M/NF=%.6f\n', ...
        buildSeconds,sys.mass.info.M,sys.mass.info.MOverNF);

    rng(20260829,'twister');
    x=to_gpu(randn(sys.ndof,1)+1i*randn(sys.ndof,1),gpuInfo.precision);
    y=to_gpu(randn(sys.ndof,1)+1i*randn(sys.ndof,1),gpuInfo.precision);
    Mx=sys.mass.Mop(x); My=sys.mass.Mop(y);
    herm=abs(gather_scalar(x'*My)-conj(gather_scalar(y'*Mx))) / ...
        max(1,abs(gather_scalar(x'*My))+abs(gather_scalar(y'*Mx)));
    rq=real(gather_scalar(x'*Mx))/real(gather_scalar(x'*x));
    fprintf('  mass Hermitian defect=%.3e, Rayleigh quotient=%.6e\n',herm,rq);
    if herm>1e-10 || ~(isfinite(rq) && rq>0)
        error('N=5 mass backend preflight failed.');
    end
    clear x y Mx My

    [Aop,getStats]=make_krinv_instrumented(sys,cfg.solver.tolCG,cfg.solver.maxitInnerCG);
    b=to_gpu(randn(sys.nr,1)+1i*randn(sys.nr,1),gpuInfo.precision);
    fprintf('[Exp3 N=5 preflight] one explicit K_r^{-1} action...\n');
    y=Aop(b); wait_for_gpu(y);
    st=getStats();
    fprintf('  action %.3f s; Mhat CG iter=%d, relres=%.3e\n', ...
        st.totalActionSeconds,st.iterations(end),st.relativeResiduals(end));
    fprintf('  inner CG time %.3f s; inverse-action other %.3f s\n', ...
        st.totalInnerCGSeconds,st.totalInverseActionOtherSeconds);

    Ng=(2*cfg.yee.n+3)^3;
    fprintf('  reconstruction target: n=%d, h=%.5f, Ng=%d per component\n', ...
        cfg.yee.n,cfg.yee.h,Ng);
    fprintf('  recovery implementation: columnwise M*x_j reuse; no 10-RHS rank-1 FFT\n');
    fprintf('[Exp3 N=5 preflight] COMPLETE.\n');
end
