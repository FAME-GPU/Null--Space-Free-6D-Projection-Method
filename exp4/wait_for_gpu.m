function wait_for_gpu(x)
%WAIT_FOR_GPU Synchronize the active GPU in the strict Exp4 production path.
    if nargin>0 && ~isa(x,'gpuArray')
        error('wait_for_gpu received a non-GPU numerical result in strict GPU mode.');
    end
    if exist('gpuDeviceCount','file')==0 || gpuDeviceCount<1
        error('A supported GPU and Parallel Computing Toolbox are required.');
    end
    wait(gpuDevice);
end
