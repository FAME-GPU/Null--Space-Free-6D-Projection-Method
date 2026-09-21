function y = applyMij_fft(x,Eij,sz,n)
%APPLYMIJ_FFT Matrix-free scalar Fourier multiplication operator.
%
%   y = F( Eij .* F^{-1}(x) ).
%
% x can be an n-by-1 vector or an n-by-nrhs matrix. For matrix input,
% each column is processed independently.

    if isvector(x)
        X = reshape(x, sz);
        yGrid = fftn(Eij .* ifftn(X));
        y = yGrid(:);
        return;
    end

    nrhs = size(x,2);
    y = complex(zeros(n,nrhs,'like',x));

    for j = 1:nrhs
        X = reshape(x(:,j),sz);
        yGrid = fftn(Eij .* ifftn(X));
        y(:,j) = yGrid(:);
    end
end
