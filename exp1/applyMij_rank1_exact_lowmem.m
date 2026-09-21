function y = applyMij_rank1_exact_lowmem(x,kernelFFT,plan)
%APPLYMIJ_RANK1_EXACT_LOWMEM Exact scalar rank-1 mass action, low-memory path.
%
% Columns are processed one at a time.  The embedded work vector is reused
% across FFT -> pointwise multiplication -> IFFT assignments, which avoids
% the multi-RHS M-by-nrhs temporaries used by the earlier batched path.

    if isvector(x), x = x(:); end
    if size(x,1) ~= plan.NF
        error('applyMij_rank1_exact_lowmem:BadSize','x must have plan.NF rows.');
    end
    nrhs = size(x,2);
    y = zeros(plan.NF,nrhs,'like',x);
    for j = 1:nrhs
        w = zeros(plan.M,1,'like',x);
        w(plan.inputIdx) = x(:,j);
        w = fft(w,[],1);
        w = w .* kernelFFT;
        w = ifft(w,[],1);
        y(:,j) = w(plan.targetIdx);
        clear w
    end
end
