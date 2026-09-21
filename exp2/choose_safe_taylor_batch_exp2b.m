function info = choose_safe_taylor_batch_exp2b(NF,requested,prototype,cfg)
%CHOOSE_SAFE_TAYLOR_BATCH_EXP2B Conservative term-batch memory guard.
%
% Qb construction can create multiple NF-by-b real temporaries.  We budget
% four such arrays plus a small fixed allowance, then enforce the same live
% GPU reserve used by the direct method.

    if ~isa(prototype,'gpuArray')
        error('choose_safe_taylor_batch_exp2b:GPURequired','prototype must be on GPU.');
    end
    dev = gpuDevice();
    avail = double(dev.AvailableMemory);
    reserve = double(cfg.memory.reserveGiB)*2^30;
    sf = double(cfg.memory.taylorWorkspaceSafetyFactor);
    cls = classUnderlying(prototype);
    if strcmp(cls,'single'), br=4; else, br=8; end

    candidates = unique([requested 24 16 12 8 4 2 1],'stable');
    candidates = candidates(candidates>0 & candidates<=requested);
    chosen = NaN;
    estimated = NaN;
    fixedAllowance = 128*2^20;
    for b = candidates
        raw = 4*double(NF)*double(b)*br + fixedAllowance;
        if sf*raw + reserve <= avail
            chosen = b;
            estimated = raw;
            break;
        end
    end
    if ~isfinite(chosen)
        error('choose_safe_taylor_batch_exp2b:InsufficientGPUMemory', ...
            'Unable to fit even a one-term Taylor batch within the configured GPU reserve.');
    end

    info = struct('termBatchSize',chosen, ...
        'estimatedRawWorkspaceBytes',estimated, ...
        'estimatedSafetyWorkspaceBytes',sf*estimated, ...
        'availableBytes',avail,'reserveBytes',reserve,'precision',cls);
end
