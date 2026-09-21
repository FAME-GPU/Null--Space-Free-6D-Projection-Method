function Y = applyM_cols12_tensor_scalar(x12,E,sz,n)
    if isvector(x12), x12=x12(:); end
    nrhs=size(x12,2);
    Y=complex(zeros(3*n,nrhs,'like',x12));
    for r=1:nrhs
        Y(1:n,r)=applyMij_fft(x12(1:n,r),E,sz,n);
        Y(n+1:2*n,r)=applyMij_fft(x12(n+1:2*n,r),E,sz,n);
    end
end
