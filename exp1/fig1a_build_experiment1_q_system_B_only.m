function sys = fig1a_build_experiment1_q_system_B_only(common,qPhysCPU,cfg)
%BUILD_EXPERIMENT1_Q_SYSTEM_B_ONLY Minimal reduced data for nested Method B.
%
% Method B needs V, sigma_r and the common mass operator to apply
%   K_r z = V^* M^{-1} V z.
% It does not need the W/Mhat/Br structures used only by explicit Method C.

    P=common.P; n=common.NF;
    qPhys=to_gpu(qPhysCPU(:),cfg.gpu.precision);
    kLift=P*((P.'*P)\qPhys);

    [Kop,U0Hop,D0vec,Vdata,sigmaR]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,common.fourier.Xi,common.fourier.sz,n,cfg.problem.dim);

    sys=struct();
    sys.N=common.N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.epsMin=common.epsMin; sys.epsMax=common.epsMax;
    sys.P=P; sys.qPhys=qPhys; sys.kLift=kLift;
    sys.mass=common.mass; sys.Kop=Kop; sys.U0Hop=U0Hop; sys.D0vec=D0vec;
    sys.Vdata=Vdata; sys.sigmaR=sigmaR;
end
