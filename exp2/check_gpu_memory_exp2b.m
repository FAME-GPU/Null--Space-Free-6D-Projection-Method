function info = check_gpu_memory_exp2b(cfg,label)
%CHECK_GPU_MEMORY_EXP2B Lightweight live GPU memory diagnostic.
    if nargin<2, label=''; end
    dev = gpuDevice();
    avail = double(dev.AvailableMemory);
    total = double(dev.TotalMemory);
    minBytes = double(cfg.memory.minAvailableGiBForRun)*2^30;
    if avail < minBytes
        error('check_gpu_memory_exp2b:LowAvailableMemory', ...
            'Available GPU memory %.2f GiB is below required minimum %.2f GiB (%s).', ...
            avail/2^30,cfg.memory.minAvailableGiBForRun,label);
    end
    info = struct('label',label,'availableBytes',avail,'totalBytes',total, ...
        'availableGiB',avail/2^30,'totalGiB',total/2^30);
end
