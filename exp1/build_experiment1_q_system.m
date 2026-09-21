function sys = build_experiment1_q_system(common,qPhysCPU,cfg)
%BUILD_EXPERIMENT1_Q_SYSTEM Build q-dependent null-space-free operators.

    P=common.P;
    n=common.NF;
    qPhys=to_gpu(qPhysCPU(:),cfg.gpu.precision);
    kLift=P*((P.'*P)\qPhys);

    [Kop,U0Hop,D0vec,Vdata,sigmaR]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,common.fourier.Xi,common.fourier.sz,n,cfg.problem.dim);

    V11=Vdata.V11; V12=Vdata.V12;
    V21=Vdata.V21; V22=Vdata.V22;
    V31=Vdata.V31; V32=Vdata.V32;
    detV1=V11.*V22-V12.*V21;
    minAbsDetV1=gather_scalar(min(abs(detV1)));
    if minAbsDetV1<1e-12
        warning('Some V1 blocks are nearly singular: min|det(V1)|=%.3e.',minAbsDetV1);
    end

    Wcoef=solveV1H_blockdiag([conj(V31);conj(V32)], ...
        V11,V12,V21,V22,detV1,n);
    W1=Wcoef(1:n); W2=Wcoef(n+1:2*n);
    [Wop,WHop]=build_W_operators(W1,W2,n);

    inner=build_inner_system_operators( ...
        common.mass,Wop,WHop,W1,W2,V11,V12,V21,V22,detV1,n);

    sys=struct();
    sys.N=common.N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.epsMin=common.epsMin; sys.epsMax=common.epsMax;
    sys.P=P; sys.qPhys=qPhys; sys.kLift=kLift;
    sys.fourier=common.fourier; sys.mass=common.mass;
    sys.Kop=Kop; sys.U0Hop=U0Hop; sys.D0vec=D0vec;
    sys.Vdata=Vdata; sys.sigmaR=sigmaR;
    sys.V11=V11; sys.V12=V12; sys.V21=V21; sys.V22=V22;
    sys.detV1=detV1; sys.W1=W1; sys.W2=W2; sys.Wop=Wop; sys.WHop=WHop;
    sys.Sop=inner.Sop; sys.MhatOp=inner.MhatOp;
    sys.BrOp=inner.BrOp; sys.BrHop=inner.BrHop;
    sys.invCorthDiag=inner.invCorthDiag; sys.KorthDiag=inner.KorthDiag;
    sys.minAbsDetV1=minAbsDetV1;
end
