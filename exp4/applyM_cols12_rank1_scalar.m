function Y = applyM_cols12_rank1_scalar(x12,kernelFFT,plan)
%APPLYM_COLS12_RANK1_SCALAR Apply diag(M,M,0) to two component columns.
    if isvector(x12), x12=x12(:); end
    n=plan.NF;
    nrhs=size(x12,2);
    if size(x12,1)~=2*n, error('x12 must have 2*NF rows.'); end
    X=rank1_exact_fft_scatter([x12(1:n,:),x12(n+1:2*n,:)],plan);
    y=rank1_exact_ifft_gather(kernelFFT.*X,plan);
    Y=complex(zeros(3*n,nrhs,'like',x12));
    Y(1:n,:)=y(:,1:nrhs);
    Y(n+1:2*n,:)=y(:,nrhs+1:2*nrhs);
end
