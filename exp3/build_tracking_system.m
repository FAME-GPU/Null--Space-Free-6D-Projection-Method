function sys = build_tracking_system(cfg,N,gpuInfo)
%BUILD_TRACKING_SYSTEM Build the null-space-free Maxwell system for tracking.

    precision=gpuInfo.precision;
    P=to_gpu(cfg.problem.P,precision);
    T=to_gpu(cfg.problem.T(:),precision);

    fourier=build_fourier_grid_6d(N,cfg.problem.dim,T,P(1));
    n=fourier.n;

    coord1=cast((0:fourier.nGrid-1)/fourier.nGrid*2*pi,'like',P(1));
    [x1,x2,x3,x4,x5,x6]=ndgrid(coord1,coord1,coord1,coord1,coord1,coord1);
    material=build_smooth_scalar_material_compact( ...
        cfg.material,x1,x2,x3,x4,x5,x6);
    clear x1 x2 x3 x4 x5 x6 coord1

    mass=build_mass_backend_exp3(material,fourier,cfg.mass,gpuInfo);
    clear material

    qPhys=to_gpu(cfg.problem.q(:),precision);
    kLift=P*((P.'*P)\qPhys);
    [Kop,U0Hop,D0vec,Vdata,sigmaR]=assemble_K_and_nullspace_free_basis( ...
        P,kLift,fourier.Xi,fourier.sz,n,cfg.problem.dim);

    V11=Vdata.V11; V12=Vdata.V12; V21=Vdata.V21; V22=Vdata.V22;
    V31=Vdata.V31; V32=Vdata.V32;
    detV1=V11.*V22-V12.*V21;

    Wcoef=solveV1H_blockdiag([conj(V31);conj(V32)], ...
        V11,V12,V21,V22,detV1,n);
    W1=Wcoef(1:n); W2=Wcoef(n+1:2*n);
    [Wop,WHop]=build_W_operators(W1,W2,n);
    inner=build_inner_system_operators( ...
        mass,Wop,WHop,W1,W2,V11,V12,V21,V22,detV1,n);

    sys=struct();
    sys.N=N; sys.NF=n; sys.nr=2*n; sys.ndof=3*n;
    sys.fourier=fourier; sys.P=P; sys.qPhys=qPhys; sys.kLift=kLift;
    sys.mass=mass; sys.Kop=Kop; sys.U0Hop=U0Hop; sys.D0vec=D0vec;
    sys.Vdata=Vdata; sys.sigmaR=sigmaR;
    sys.V11=V11; sys.V12=V12; sys.V21=V21; sys.V22=V22;
    sys.detV1=detV1; sys.W1=W1; sys.W2=W2;
    sys.Wop=Wop; sys.WHop=WHop;
    sys.Sop=inner.Sop; sys.MhatOp=inner.MhatOp;
    sys.BrOp=inner.BrOp; sys.BrHop=inner.BrHop;
    sys.invCorthDiag=inner.invCorthDiag; sys.KorthDiag=inner.KorthDiag;

    if cfg.output.verbose
        fprintf('Built tracking system N=%d: NF=%d, reduced dim=%d, rank1 M=%d.\n', ...
            N,n,2*n,mass.info.M);
    end
end
