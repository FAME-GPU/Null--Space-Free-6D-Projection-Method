function gpuInfo = reset_gpu_for_scale(gpuCfg,N,method)
%RESET_GPU_FOR_SCALE Start each N from a clean GPU allocator state.
    g=gpuDevice(gpuCfg.deviceIndex);
    if isfield(gpuCfg,'resetBetweenN') && gpuCfg.resetBetweenN
        reset(g); g=gpuDevice(gpuCfg.deviceIndex);
    end
    opt=gpuCfg; opt.resetDevice=false;
    gpuInfo=gpu_initialize(opt);
    fprintf('[Fig1bc %s] N=%d clean available GPU memory = %.3f GiB\n', ...
        method,N,g.AvailableMemory/2^30);
end
