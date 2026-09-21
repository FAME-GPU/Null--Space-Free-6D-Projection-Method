function sys = build_experiment1_q_system_B_only(common,qPhysCPU,cfg)
%BUILD_EXPERIMENT1_Q_SYSTEM_B_ONLY Minimal reduced data for nested Method B.
% Keeps only V, sigma_r and the common mass operator.

    P=common.P; n=common.NF;
    qPhys=to_gpu(qPhysCPU(:),cfg.gpu.precision);
    kLift=P*((P.'*P)\qPhys);
    [~,~,~,Vdata,sigmaR]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,common.fourier.Xi,common.fourier.sz,n,cfg.problem.dim);

    sys=struct();
    sys.N=common.N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.mass=common.mass; sys.Vdata=Vdata; sys.sigmaR=sigmaR;
end
