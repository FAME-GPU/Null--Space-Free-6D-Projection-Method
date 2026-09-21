function common = fig1a_build_experiment1_common(cfg,gpuInfo,N)
%BUILD_EXPERIMENT1_COMMON Build N-dependent objects shared by all Bloch points.
%
% The Fourier grid, scalar material, exact rank-1 plan and its single scalar
% kernel do not depend on q and are therefore built exactly once and reused
% over all formal Bloch problems.

    precision=gpuInfo.precision;
    P=to_gpu(cfg.problem.P,precision);
    T=to_gpu(cfg.problem.T(:),precision);
    fourier=build_fourier_grid_6d(N,cfg.problem.dim,T,P(1));

    coord1=cast((0:fourier.nGrid-1)/fourier.nGrid*2*pi,'like',P(1));
    [x1,x2,x3,x4,x5,x6]=ndgrid(coord1,coord1,coord1,coord1,coord1,coord1);
    material=build_smooth_scalar_material_compact(cfg.material,x1,x2,x3,x4,x5,x6);
    clear x1 x2 x3 x4 x5 x6 coord1

    epsMin=gather_scalar(min(real(material.scalarField(:))));
    epsMax=gather_scalar(max(real(material.scalarField(:))));
    epsMean=gather_scalar(mean(real(material.scalarField(:))));
    mass=fig1a_build_mass_backend_large_scalar(material,fourier,cfg.mass,gpuInfo);
    % The rank-1 backend retains its convolution kernel, not the sampled field.
    % Release the persistent material array after kernel construction.
    material.scalarField=[];

    common=struct();
    common.N=N;
    common.NF=fourier.n;
    common.nr=2*fourier.n;
    common.ndof=3*fourier.n;
    common.P=P;
    common.T=T;
    common.fourier=fourier;
    common.material=material;
    common.mass=mass;
    common.epsMin=epsMin;
    common.epsMax=epsMax;
    common.epsMean=epsMean;
    common.kappaM=epsMax/epsMin;
end
