function Y = applyM_cols12_rank1_scalar_lowmem(x12,kernelFFT,plan)
%APPLYM_COLS12_RANK1_SCALAR_LOWMEM Serial exact diag(M,M,0) action.
    if isvector(x12), x12=x12(:); end
    n=plan.NF; nrhs=size(x12,2);
    if size(x12,1)~=2*n, error('x12 must have 2*NF rows.'); end
    Y=zeros(3*n,nrhs,'like',x12);
    for j=1:nrhs
        Y(1:n,j)=applyMij_rank1_exact_lowmem(x12(1:n,j),kernelFFT,plan);
        Y(n+1:2*n,j)=applyMij_rank1_exact_lowmem(x12(n+1:2*n,j),kernelFFT,plan);
    end
end
