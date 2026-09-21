function mass = build_mass_backend_exp3(mat,fourier,massCfg,gpuInfo)
%BUILD_MASS_BACKEND_EXP3 Strict-GPU exact scalar rank-1 mass backend.
    backend=lower(char(massCfg.backend));
    if ~ismember(backend,{'rank1-exact-scalar','rank1-exact','exact-rank1','rank1'})
        error('This Exp4 package permits only the exact GPU scalar rank-1 backend.');
    end
    E=mat.scalarField;
    if ~isa(E,'gpuArray')
        error('Mass material field must be a gpuArray.');
    end

    plan=build_radix_exact_rank1_plan_scalar( ...
        fourier.N,fourier.dim,gpuInfo.precision,true);

    key=sprintf('N%d',fourier.N);
    if isfield(massCfg,'rank1') && isfield(massCfg.rank1,key)
        pcfg=massCfg.rank1.(key);
        if isfield(pcfg,'M') && double(pcfg.M)~=double(plan.M)
            error('Configured rank-1 M does not match radix exact plan for N=%d.',fourier.N);
        end
        if isfield(pcfg,'z') && any(double(pcfg.z(:))~=double(plan.z(:)))
            error('Configured rank-1 z does not match radix exact plan for N=%d.',fourier.N);
        end
    end

    kernelFFT=build_exact_rank1_scalar_kernel_fft(E,plan);
    if ~isa(kernelFFT,'gpuArray')
        error('Rank-1 kernel unexpectedly left the GPU.');
    end

    mass.name='rank1-exact-scalar-gpu';
    mass.Mop=@(U)applyM_component_first_rank1_scalar(U,kernelFFT,plan);
    mass.Mcol3op=@(x)applyM_col3_rank1_scalar(x,kernelFFT,plan);
    mass.Mcols12op=@(x12)applyM_cols12_rank1_scalar(x12,kernelFFT,plan);
    mass.MscalarOp=@(x)applyMij_rank1_exact(x,kernelFFT,plan);
    mass.info=struct('name',mass.name,'N',fourier.N,'NF',plan.NF, ...
        'M',plan.M,'MOverNF',plan.embeddingRatio,'z',plan.z(:).', ...
        'exactCertified',true,'scalarOptimized',true, ...
        'kernelStorageCount',1,'B',plan.B,'device','GPU');
    mass.plan=plan;
end
