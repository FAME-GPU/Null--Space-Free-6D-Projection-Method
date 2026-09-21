function MU = applyM_component_first_rank1_scalar_lowmem(U,kernelFFT,plan)
%APPLYM_COMPONENT_FIRST_RANK1_SCALAR_LOWMEM Serial exact scalar-isotropic M action.
    if isvector(U), U=U(:); end
    n=plan.NF; nrhs=size(U,2);
    if size(U,1)~=3*n, error('U must have 3*NF rows.'); end
    MU=zeros(3*n,nrhs,'like',U);
    for j=1:nrhs
        MU(1:n,j)=applyMij_rank1_exact_lowmem(U(1:n,j),kernelFFT,plan);
        MU(n+1:2*n,j)=applyMij_rank1_exact_lowmem(U(n+1:2*n,j),kernelFFT,plan);
        MU(2*n+1:3*n,j)=applyMij_rank1_exact_lowmem(U(2*n+1:3*n,j),kernelFFT,plan);
    end
end
