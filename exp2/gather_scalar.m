function y = gather_scalar(x)
%GATHER_SCALAR Gather a scalar GPU value and return a CPU double/single.

    if isa(x,'gpuArray')
        y = gather(x);
    else
        y = x;
    end

    if ~isscalar(y)
        error('gather_scalar expects a scalar input.');
    end
end
