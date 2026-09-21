function kernelFFT = build_exact_rank1_scalar_kernel_fft(E,plan)
%BUILD_EXACT_RANK1_SCALAR_KERNEL_FFT Build the one scalar exact kernel.
%
% For the radix plan every 1-D kernel location 0,...,M-1 is active.  Its
% tensor circular residue can be decoded directly from the radix-B digits,
% avoiding the O(B^dim log B^dim) generic collision sort and avoiding a
% persistent M-entry residue-index array.

    Ehat=fftn(E);
    EhatVec=Ehat(:);
    h=zeros(plan.M,1,'like',EhatVec);

    chunk=plan.kernelChunkSize;
    nGrid=plan.nGrid;
    B=plan.B;
    s=nGrid-1;
    dim=plan.dim;
    M=plan.M;

    for first0=0:chunk:M-1
        last0=min(M-1,first0+chunk-1);
        code=(first0:last0).';
        tmp=code;
        residueLinear=zeros(size(code));
        stride=1;
        for j=1:dim
            digit=mod(tmp,B);
            tmp=floor(tmp/B);
            residueDigit=mod(digit-s,nGrid);
            residueLinear=residueLinear+residueDigit*stride;
            stride=stride*nGrid;
        end
        idx=uint32(residueLinear+1);
        h(first0+1:last0+1)=EhatVec(idx)/plan.NF;
    end

    kernelFFT=fft(h,[],1);
end
