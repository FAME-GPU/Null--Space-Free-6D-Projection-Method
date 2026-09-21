function Y = applyM_col3_tensor_scalar(x,E,sz,n)
    if isvector(x), x=x(:); end
    nrhs=size(x,2);
    Y=complex(zeros(3*n,nrhs,'like',x));
    for r=1:nrhs
        Y(2*n+1:3*n,r)=applyMij_fft(x(:,r),E,sz,n);
    end
end
