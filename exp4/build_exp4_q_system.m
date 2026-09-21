function sys = build_exp4_q_system(st,qCPU,cfg)
%BUILD_EXP4_Q_SYSTEM Build the q-dependent null-space-free operators.
    qPhys=to_gpu(qCPU(:),st.gpuInfo.precision);
    P=st.P;
    kLift=P*((P.'*P)\qPhys);
    [Kop,U0Hop,D0vec,Vdata,sigmaR]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,st.fourier.Xi,st.fourier.sz,st.NF,cfg.problem.dim);

    n=st.NF;
    V11=Vdata.V11; V12=Vdata.V12; V21=Vdata.V21; V22=Vdata.V22;
    V31=Vdata.V31; V32=Vdata.V32;
    detV1=V11.*V22-V12.*V21;

    Wcoef=solveV1H_blockdiag([conj(V31);conj(V32)], ...
        V11,V12,V21,V22,detV1,n);
    W1=Wcoef(1:n); W2=Wcoef(n+1:2*n);
    [Wop,WHop]=build_W_operators(W1,W2,n);
    inner=build_inner_system_operators( ...
        st.mass,Wop,WHop,W1,W2,V11,V12,V21,V22,detV1,n);

    sys=struct();
    sys.N=cfg.problem.N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.P=P; sys.qPhys=qPhys; sys.kLift=kLift; sys.fourier=st.fourier;
    sys.mass=st.mass; sys.Kop=Kop; sys.U0Hop=U0Hop; sys.D0vec=D0vec;
    sys.Vdata=Vdata; sys.sigmaR=sigmaR;
    sys.V11=V11; sys.V12=V12; sys.V21=V21; sys.V22=V22;
    sys.detV1=detV1; sys.W1=W1; sys.W2=W2;
    sys.Wop=Wop; sys.WHop=WHop;
    sys.Sop=inner.Sop; sys.MhatOp=inner.MhatOp;
    sys.BrOp=inner.BrOp; sys.BrHop=inner.BrHop;
    sys.invCorthDiag=inner.invCorthDiag; sys.KorthDiag=inner.KorthDiag;
end
