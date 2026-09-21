function MU = applyM_component_first_rank1_scalar(U,kernelFFT,plan)
%APPLYM_COMPONENT_FIRST_RANK1_SCALAR Exact scalar-isotropic 3-component mass action.

    if isvector(U), U=U(:); end
    n=plan.NF;
    nrhs=size(U,2);
    if size(U,1)~=3*n
        error('U must have 3*NF rows.');
    end

    U1=U(1:n,:); U2=U(n+1:2*n,:); U3=U(2*n+1:3*n,:);
    X=rank1_exact_fft_scatter([U1,U2,U3],plan);
    Y=kernelFFT.*X;
    y=rank1_exact_ifft_gather(Y,plan);

    MU=complex(zeros(3*n,nrhs,'like',U));
    MU(1:n,:)=y(:,1:nrhs);
    MU(n+1:2*n,:)=y(:,nrhs+1:2*nrhs);
    MU(2*n+1:3*n,:)=y(:,2*nrhs+1:3*nrhs);
end
