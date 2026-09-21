function y = to_cpu(x)
%TO_CPU Gather gpuArray data; leave ordinary MATLAB arrays unchanged.
    if isa(x,'gpuArray')
        y=gather(x);
    else
        y=x;
    end
end
