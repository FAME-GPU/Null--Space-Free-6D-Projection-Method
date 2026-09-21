function wait_for_gpu(x)
%WAIT_FOR_GPU Synchronize the active GPU.
% If an input is supplied, synchronize only when it is a gpuArray.

    if nargin==0
        wait(gpuDevice);
        return;
    end
    if isa(x,'gpuArray')
        wait(gpuDevice);
    end
end
