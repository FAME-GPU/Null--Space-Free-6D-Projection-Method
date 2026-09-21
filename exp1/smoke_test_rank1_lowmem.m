function smoke_test_rank1_lowmem()
%SMOKE_TEST_RANK1_LOWMEM GPU preflight for rank-1 and compact Method C.
    cfg=make_config_exp1bc();
    opt=cfg.gpu; opt.resetDevice=true; g=gpu_initialize(opt);
    N=2; P=to_gpu(cfg.problem.P,g.precision); T=to_gpu(cfg.problem.T(:),g.precision);
    fourier=build_fourier_grid_6d(N,cfg.problem.dim,T,P(1));
    coord1=cast((0:fourier.nGrid-1)/fourier.nGrid*2*pi,'like',P(1));
    [x1,x2,x3,x4,x5,x6]=ndgrid(coord1,coord1,coord1,coord1,coord1,coord1);
    mat=build_smooth_scalar_material_compact(cfg.material,x1,x2,x3,x4,x5,x6);
    clear x1 x2 x3 x4 x5 x6 coord1

    rcfg=struct('backend','rank1-exact-scalar-lowmem','rank1',struct());
    tcfg=struct('backend','batchfft-scalar','rank1',struct());
    mr=build_mass_backend_large_scalar(mat,fourier,rcfg,g);
    mt=build_mass_backend_large_scalar(mat,fourier,tcfg,g);
    rng(20260909,'twister');
    x=to_gpu(randn(3*fourier.n,1)+1i*randn(3*fourier.n,1),g.precision);
    yr=mr.Mop(x); yt=mt.Mop(x); wait_for_gpu(yr); wait_for_gpu(yt);
    relMass=gather_scalar(norm(yr-yt,2))/max(gather_scalar(norm(yt,2)),realmin);
    fprintf('Low-memory rank-1 vs tensor mass relative error: %.3e\n',relMass);
    if ~(isfinite(relMass) && relMass<1e-11)
        error('EXP1:Rank1SmokeFailed','Rank-1 mass smoke test failed: %.3e.',relMass);
    end

    common=struct('N',N,'NF',fourier.n,'P',P,'fourier',fourier,'mass',mr, ...
        'epsMin',gather_scalar(min(mat.scalarField(:))), ...
        'epsMax',gather_scalar(max(mat.scalarField(:))));
    sysG=build_experiment1_q_system(common,cfg.scale.qPhys,cfg);
    sysC=build_experiment1_q_system_C_compact(common,cfg.scale.qPhys,cfg);
    bInf=make_run_budget(Inf,Inf);
    [opG,~]=make_explicit_krinv_instrumented(sysG,cfg.explicit.tolMhat,cfg.explicit.maxitMhat,bInf);
    [opC,~]=make_explicit_krinv_compact(sysC,cfg.explicit.tolMhat,cfg.explicit.maxitMhat,bInf);
    q=to_gpu(randn(2*fourier.n,1)+1i*randn(2*fourier.n,1),g.precision);
    yg=opG(q); yc=opC(q); wait_for_gpu(yg); wait_for_gpu(yc);
    relC=gather_scalar(norm(yg-yc,2))/max(gather_scalar(norm(yg,2)),realmin);
    fprintf('Compact C inverse vs generic C inverse relative error: %.3e\n',relC);
    if ~(isfinite(relC) && relC<1e-9)
        error('EXP1:CCompactSmokeFailed','Compact C inverse smoke test failed: %.3e.',relC);
    end

    % Independent hybrid-Lanczos sanity test on a known Hermitian diagonal
    % operator. This checks the new CPU-basis/GPU-operator recurrence before
    % any large Maxwell run.
    nToy=80; kToy=5; pToy=24;
    dToy=linspace(100,1,nToy).'; dToyGPU=to_gpu(dToy,g.precision);
    rng(20260910,'twister'); vToy=randn(nToy,1)+1i*randn(nToy,1);
    [thetaToy,~,infoToy]=Lanczos_hybrid_identity_gpuop( ...
        @(x)dToyGPU.*x,nToy,kToy,1e-10,40,pToy,vToy,g.precision,false,make_run_budget(Inf,Inf));
    exactToy=sort(dToy,'descend'); exactToy=exactToy(1:kToy);
    relToy=norm(sort(thetaToy,'descend')-exactToy,2)/norm(exactToy,2);
    fprintf('Hybrid Lanczos diagonal-operator relative eigenvalue error: %.3e\n',relToy);
    if ~infoToy.converged || ~(isfinite(relToy) && relToy<1e-7)
        error('EXP1:HybridLanczosSmokeFailed','Hybrid Lanczos smoke test failed: %.3e.',relToy);
    end

    fprintf('GPU smoke tests PASSED.\n');
end
