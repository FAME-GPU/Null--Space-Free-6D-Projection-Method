function mass = fig1a_build_mass_backend_large_scalar(mat,fourier,massCfg,gpuInfo)
%BUILD_MASS_BACKEND_LARGE_SCALAR Scalar-isotropic Fourier mass backend.
%
% Production: 'rank1-exact-scalar' uses the analytical radix exact plan and
% stores only one scalar convolution kernel.  'batchfft-scalar' is retained
% as an independent small-N preflight reference.

    backend=lower(char(massCfg.backend));
    E=mat.scalarField;
    n=fourier.n;
    sz=fourier.sz;

    switch backend
        case {'rank1-exact-scalar','rank1-exact','exact-rank1','rank1'}
            useGPU=true;
            if isfield(gpuInfo,'useGPU'), useGPU=gpuInfo.useGPU; end
            plan=build_radix_exact_rank1_plan_scalar( ...
                fourier.N,fourier.dim,gpuInfo.precision,useGPU);

            % Optional configuration guard: if M/z are supplied, require the
            % analytical radix values to agree exactly with the configuration.
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

            kernelFFT=fig1a_build_exact_rank1_scalar_kernel_fft(E,plan);
            mass.name='rank1-exact-scalar';
            mass.Mop=@(U)applyM_component_first_rank1_scalar(U,kernelFFT,plan);
            mass.Mcol3op=@(x)applyM_col3_rank1_scalar(x,kernelFFT,plan);
            mass.Mcols12op=@(x12)applyM_cols12_rank1_scalar(x12,kernelFFT,plan);
            mass.MscalarOp=@(x)applyMij_rank1_exact(x,kernelFFT,plan);
            mass.info=struct('name',mass.name,'N',fourier.N,'NF',plan.NF, ...
                'M',plan.M,'MOverNF',plan.embeddingRatio,'z',plan.z(:).', ...
                'exactCertified',true,'scalarOptimized',true, ...
                'kernelStorageCount',1,'B',plan.B);
            mass.plan=plan;

        case {'batchfft-scalar','batchfft'}
            mass.name='batchfft-scalar';
            mass.Mop=@(U)applyM_component_first_tensor_scalar(U,E,sz,n);
            mass.Mcol3op=@(x)applyM_col3_tensor_scalar(x,E,sz,n);
            mass.Mcols12op=@(x12)applyM_cols12_tensor_scalar(x12,E,sz,n);
            mass.MscalarOp=@(x)applyMij_fft(x,E,sz,n);
            mass.info=struct('name',mass.name,'N',fourier.N,'NF',n, ...
                'M',n,'MOverNF',1,'z',[],'exactCertified',true, ...
                'scalarOptimized',true,'kernelStorageCount',0,'B',NaN);

        otherwise
            error('Unknown scalar mass backend: %s',backend);
    end
end
