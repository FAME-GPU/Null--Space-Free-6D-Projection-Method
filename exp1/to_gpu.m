function y = to_gpu(x,precision)
%TO_GPU Cast a numeric array and transfer it to the current GPU.

    if nargin < 2 || isempty(precision)
        precision = 'double';
    end

    if isa(x,'gpuArray')
        if strcmpi(classUnderlying(x),precision)
            y = x;
        else
            y = cast(x,precision);
        end
    else
        y = gpuArray(cast(x,precision));
    end
end
