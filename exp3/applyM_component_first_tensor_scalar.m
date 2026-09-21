function MU = applyM_component_first_tensor_scalar(U,E,sz,n)
%APPLYM_COMPONENT_FIRST_TENSOR_SCALAR Scalar-isotropic tensor FFT reference.
    if isvector(U), U=U(:); end
    nrhs=size(U,2);
    MU=complex(zeros(size(U),'like',U));
    for r=1:nrhs
        MU(1:n,r)=applyMij_fft(U(1:n,r),E,sz,n);
        MU(n+1:2*n,r)=applyMij_fft(U(n+1:2*n,r),E,sz,n);
        MU(2*n+1:3*n,r)=applyMij_fft(U(2*n+1:3*n,r),E,sz,n);
    end
end
