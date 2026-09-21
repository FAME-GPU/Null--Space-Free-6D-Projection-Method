function info = choose_safe_direct_block_size_exp2b(NF,prototype,cfg)
%CHOOSE_SAFE_DIRECT_BLOCK_SIZE_EXP2B Pick the largest safe dense-phase batch.
%
% The direct kernel simultaneously holds a real phase matrix and a complex
% exponential matrix.  For double precision this is approximately
% 24*NF*blockSize bytes.  A safety factor and a fixed free-memory reserve
% are enforced using live gpuDevice.AvailableMemory.

    if ~isa(prototype,'gpuArray')
        error('choose_safe_direct_block_size_exp2b:GPURequired','prototype must be on GPU.');
    end
    dev = gpuDevice();
    avail = double(dev.AvailableMemory);
    reserve = double(cfg.memory.reserveGiB)*2^30;
    sf = double(cfg.memory.directWorkspaceSafetyFactor);

    cls = classUnderlying(prototype);
    if strcmp(cls,'single')
        br = 4;
    else
        br = 8;
    end
    bc = 2*br;

    candidates = unique(round(cfg.scaling.directBlockCandidates(:).'),'stable');
    candidates = candidates(candidates>0 & candidates<=cfg.scaling.directBlockSizeRequested);
    if isempty(candidates)
        error('choose_safe_direct_block_size_exp2b:NoCandidates','No positive block-size candidates are available.');
    end

    chosen = NaN;
    estimated = NaN;
    for b = candidates
        raw = double(NF)*double(b)*(br+bc);
        need = sf*raw;
        if need + reserve <= avail
            chosen = b;
            estimated = raw;
            break;
        end
    end
    if ~isfinite(chosen)
        error('choose_safe_direct_block_size_exp2b:InsufficientGPUMemory', ...
            ['No direct block size satisfies the configured memory reserve. ', ...
             'Available %.2f GiB; reserve %.2f GiB.'],avail/2^30,reserve/2^30);
    end

    info = struct();
    info.blockSize = chosen;
    info.estimatedRawWorkspaceBytes = estimated;
    info.estimatedSafetyWorkspaceBytes = sf*estimated;
    info.availableBytes = avail;
    info.reserveBytes = reserve;
    info.precision = cls;
end
