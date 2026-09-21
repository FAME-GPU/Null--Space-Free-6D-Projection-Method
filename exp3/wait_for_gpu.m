function wait_for_gpu(x)
%WAIT_FOR_GPU Synchronize the active GPU when one is actually involved.
%
% With a gpuArray input, synchronize its active device.  With no input,
% synchronize only if Parallel Computing Toolbox and at least one supported
% GPU are available.  This keeps the CPU-only reconstruction self-test usable.

    if nargin>0
        if isa(x,'gpuArray')
            wait(gpuDevice);
        end
        return;
    end

    if exist('gpuDeviceCount','file')~=0
        try
            if gpuDeviceCount>0
                wait(gpuDevice);
            end
        catch
            % No-op for CPU-only sanity checks.  Production GPU setup is
            % validated separately by gpu_initialize before Experiment 3.
        end
    end
end
