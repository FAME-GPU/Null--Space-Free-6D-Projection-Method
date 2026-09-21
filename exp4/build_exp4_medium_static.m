function st = build_exp4_medium_static(cfg,gpuInfo)
%BUILD_EXP4_MEDIUM_STATIC Build q-independent Fourier/material/mass data on GPU.
    precision=gpuInfo.precision;
    P=to_gpu(cfg.problem.P,precision);
    T=to_gpu(cfg.problem.T(:),precision);
    fourier=build_fourier_grid_6d(cfg.problem.N,cfg.problem.dim,T,P(1));

    coord1=cast((0:fourier.nGrid-1)/fourier.nGrid*2*pi,'like',P(1));
    [x1,x2,x3,x4,x5,x6]=ndgrid(coord1,coord1,coord1,coord1,coord1,coord1);
    [material,materialInfoGPU]=build_quasiperiodic_scalar_material_exp4( ...
        cfg.material,x1,x2,x3,x4,x5,x6);

    materialInfoGPU.discreteTensorMeanPermittivity=sum(material.scalarField(:))/numel(material.scalarField);
    materialInfoGPU.discreteMeanRelativeToTarget= ...
        (materialInfoGPU.discreteTensorMeanPermittivity-materialInfoGPU.targetMeanPermittivity) ...
        ./materialInfoGPU.targetMeanPermittivity;

    clear x1 x2 x3 x4 x5 x6 coord1

    wait_for_gpu(); t=tic;
    mass=build_mass_backend_exp3(material,fourier,cfg.mass,gpuInfo);
    wait_for_gpu(); buildSeconds=toc(t);
    clear material

    % Retain only the GPU scalar scale needed by Yee-space sampling.
    materialRuntime=struct();
    materialRuntime.epsScale=materialInfoGPU.epsScale;
    materialRuntime.targetMeanPermittivity=materialInfoGPU.targetMeanPermittivity;

    % Gather only reporting metadata after all material calculations finish.
    materialInfo=struct();
    fn=fieldnames(materialInfoGPU);
    for j=1:numel(fn)
        val=materialInfoGPU.(fn{j});
        if isa(val,'gpuArray'), val=gather(val); end
        materialInfo.(fn{j})=val;
    end

    st=struct();
    st.P=P; st.T=T; st.fourier=fourier; st.mass=mass;
    st.NF=fourier.n; st.nr=2*fourier.n; st.ndof=3*fourier.n;
    st.gpuInfo=gpuInfo; st.massBuildSeconds=buildSeconds;
    st.materialInfo=materialInfo;
    st.materialRuntime=materialRuntime;

    if cfg.output.verbose
        fprintf('[Exp4 %s] static GPU build: N=%d, NF=%d, reduced dim=%d, rank1 M=%d, %.2f s\n', ...
            cfg.output.tag,cfg.problem.N,st.NF,st.nr,mass.info.M,buildSeconds);
        fprintf('  material model=%s, epsScale=%.12g\n',materialInfo.modelType,materialInfo.epsScale);
        fprintf('  continuum mean eps=%.12g, target=%.12g, rel.mismatch=%.3e\n', ...
            materialInfo.continuumMeanPermittivity,materialInfo.targetMeanPermittivity, ...
            materialInfo.relativeContinuumMeanMismatch);
        fprintf('  tensor-grid mean eps=%.12g, rel.to target=%.3e\n', ...
            materialInfo.discreteTensorMeanPermittivity,materialInfo.discreteMeanRelativeToTarget);
    end
end
