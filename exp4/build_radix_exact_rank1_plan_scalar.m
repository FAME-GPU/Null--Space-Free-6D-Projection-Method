function plan = build_radix_exact_rank1_plan_scalar(N,dim,precision,useGPU)
%BUILD_RADIX_EXACT_RANK1_PLAN_SCALAR Exact radix rank-1 plan; GPU production.
    if nargin<3 || isempty(precision), precision='double'; end
    if nargin<4 || isempty(useGPU), useGPU=true; end
    if ~useGPU, error('Exp4 production rank-1 plan requires GPU execution.'); end

    nGrid=2*N;
    B=2*nGrid-1;
    NF=nGrid^dim;
    z=double(B).^(0:dim-1).';
    M=double(nGrid)*double(B)^(dim-1);
    s=nGrid-1;
    if M>=double(intmax('uint32'))
        error('Rank-1 length M=%g exceeds uint32 indexing range.',M);
    end

    % Build the million retained radix codes directly on GPU.
    idx0=gpuArray.colon(0,NF-1).';
    if strcmpi(precision,'single'), idx0=single(idx0); else, idx0=double(idx0); end
    tmp=idx0;
    inputCode=zeros(NF,1,'like',idx0);
    for j=1:dim
        kj=mod(tmp,nGrid);
        tmp=floor(tmp/nGrid);
        inputCode=inputCode+kj*z(j);
    end
    if gather_scalar(max(inputCode))>=M
        error('Unexpected retained radix code outside [0,M-1].');
    end

    shiftCode=mod(s*sum(z),M);
    targetCode=mod(inputCode+shiftCode,M);

    plan=struct();
    plan.method='universally exact radix rank-1 convolution (GPU scalar optimized)';
    plan.N=N; plan.dim=dim; plan.nGrid=nGrid;
    plan.sz=nGrid*ones(1,dim); plan.NF=NF; plan.B=B; plan.M=M;
    plan.z=z; plan.embeddingRatio=M/NF; plan.shiftCode=shiftCode;
    % Gather only integer indexing metadata; all radix arithmetic above is GPU.
    plan.inputIdx=gather(uint32(inputCode+1));
    plan.targetIdx=gather(uint32(targetCode+1));
    plan.precision=precision; plan.useGPU=true;
    plan.exactCertified=true; plan.kernelIsFull=true;
    plan.kernelChunkSize=1e6;
end
