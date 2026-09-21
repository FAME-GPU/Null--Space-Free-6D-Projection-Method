function Y = applyM_col3_rank1_scalar_lowmem(x,kernelFFT,plan)
%APPLYM_COL3_RANK1_SCALAR_LOWMEM Serial exact [0;0;M] action.
    if isvector(x), x=x(:); end
    n=plan.NF; nrhs=size(x,2);
    if size(x,1)~=n, error('x must have NF rows.'); end
    Y=zeros(3*n,nrhs,'like',x);
    for j=1:nrhs
        Y(2*n+1:3*n,j)=applyMij_rank1_exact_lowmem(x(:,j),kernelFFT,plan);
    end
end
