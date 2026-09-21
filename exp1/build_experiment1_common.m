function common = build_experiment1_common(cfg,gpuInfo,N)
%BUILD_EXPERIMENT1_COMMON Build N-dependent common rank-1/material data.
% This setup is deliberately outside the Fig. 1(c) solver timer.

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
    mass=build_mass_backend_large_scalar(material,fourier,cfg.mass,gpuInfo);
    material.scalarField=[];

    common=struct('N',N,'NF',fourier.n,'nr',2*fourier.n,'ndof',3*fourier.n, ...
        'P',P,'T',T,'fourier',fourier,'material',material,'mass',mass, ...
        'epsMin',epsMin,'epsMax',epsMax,'epsMean',epsMean,'kappaM',epsMax/epsMin);
end
