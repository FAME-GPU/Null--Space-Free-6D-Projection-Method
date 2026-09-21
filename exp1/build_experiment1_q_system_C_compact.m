function sys = build_experiment1_q_system_C_compact(common,qPhysCPU,cfg)
%BUILD_EXPERIMENT1_Q_SYSTEM_C_COMPACT Compact scalar-isotropic Method-C setup.
%
% Only the arrays needed by the explicit inverse are retained. Generic K,
% U0, V3 and padded 3-component inner operators are discarded after setup.

    P=common.P; n=common.NF;
    qPhys=to_gpu(qPhysCPU(:),cfg.gpu.precision);
    kLift=P*((P.'*P)\qPhys);
    [~,~,~,Vdata,~]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,common.fourier.Xi,common.fourier.sz,n,cfg.problem.dim);

    V11=Vdata.V11; V12=Vdata.V12; V21=Vdata.V21; V22=Vdata.V22;
    V31=Vdata.V31; V32=Vdata.V32;
    detV1=V11.*V22-V12.*V21;
    minAbsDetV1=gather_scalar(min(abs(detV1)));
    if minAbsDetV1<1e-12
        warning('Some V1 blocks are nearly singular: min|det(V1)|=%.3e.',minAbsDetV1);
    end

    rhsW=[conj(V31);conj(V32)];
    Wcoef=solveV1H_blockdiag(rhsW,V11,V12,V21,V22,detV1,n);
    clear rhsW V31 V32 Vdata
    W1=Wcoef(1:n); W2=Wcoef(n+1:2*n); clear Wcoef

    KorthDiag=real(1+abs(W1).^2+abs(W2).^2);
    minKorth=gather_scalar(min(KorthDiag));
    if ~(isfinite(minKorth) && minKorth>0)
        error('C compact setup produced a nonpositive orthogonalization diagonal.');
    end
    invCorthDiag=1./sqrt(KorthDiag); clear KorthDiag

    sys=struct();
    sys.N=common.N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.mass=common.mass;
    sys.V11=V11; sys.V12=V12; sys.V21=V21; sys.V22=V22; sys.detV1=detV1;
    sys.W1=W1; sys.W2=W2; sys.invCorthDiag=invCorthDiag;
    sys.minAbsDetV1=minAbsDetV1;
end
