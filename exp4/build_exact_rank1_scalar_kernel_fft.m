function kernelFFT = build_exact_rank1_scalar_kernel_fft(E,plan)
%BUILD_EXACT_RANK1_SCALAR_KERNEL_FFT Build exact scalar rank-1 kernel on GPU.
    if ~isa(E,'gpuArray')
        error('Exact rank-1 kernel construction must remain on GPU.');
    end

    Ehat=fftn(E);
    EhatVec=Ehat(:);
    h=complex(zeros(plan.M,1,'like',EhatVec));

    chunk=plan.kernelChunkSize;
    nGrid=plan.nGrid; B=plan.B; s=nGrid-1; dim=plan.dim; M=plan.M;
    precision=classUnderlying(EhatVec);

    for first0=0:chunk:M-1
        last0=min(M-1,first0+chunk-1);
        code=gpuArray.colon(first0,last0).';
        if strcmp(precision,'single'), code=single(code); else, code=double(code); end
        tmp=code;
        residueLinear=zeros(size(code),'like',code);
        stride=1;
        for j=1:dim
            digit=mod(tmp,B);
            tmp=floor(tmp/B);
            residueDigit=mod(digit-s,nGrid);
            residueLinear=residueLinear+residueDigit*stride;
            stride=stride*nGrid;
        end
        % Gather only integer subscripts; FFT data and kernel arithmetic stay on GPU.
        idxCPU=gather(uint32(residueLinear+1));
        h(first0+1:last0+1)=EhatVec(idxCPU)./plan.NF;
    end

    kernelFFT=fft(h,[],1);
end
