function Y = applyM_col3_rank1_scalar(x,kernelFFT,plan)
%APPLYM_COL3_RANK1_SCALAR Apply [0;0;M]x for scalar isotropic material.
    if isvector(x), x=x(:); end
    n=plan.NF;
    nrhs=size(x,2);
    if size(x,1)~=n, error('x must have NF rows.'); end
    mx=applyMij_rank1_exact(x,kernelFFT,plan);
    Y=complex(zeros(3*n,nrhs,'like',x));
    Y(2*n+1:3*n,:)=mx;
end
