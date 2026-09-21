function info = check_large_window_gpu_capacity_exp3(cfg,basis,n,h,part)
%CHECK_LARGE_WINDOW_GPU_CAPACITY_EXP3 Conservative memory preflight for L=4.
%
% IMPORTANT: this routine must never reset/reselect the GPU.  It is executed
% after the reusable Taylor basis has been built.  Calling gpuDevice(index)
% here would invalidate those gpuArray objects.  gpuDevice with no argument
% only queries the active device and preserves existing device data.
%
% The available-memory measurement therefore already excludes the resident
% Taylor basis.  The estimate below concerns the additional field,
% reconstruction, Yee and LWRQ workspace needed for one tracked mode.

    assert_gpu_basis_alive_exp3(basis,'Large-window preflight (entry)');

    g = gpuDevice;
    if isprop(g,'Index') && g.Index ~= cfg.gpu.deviceIndex
        error(['Large-window preflight is running on GPU %d, but the ', ...
               'configuration requests GPU %d.  Refusing to switch devices ', ...
               'because doing so would invalidate the resident Taylor basis.'], ...
               g.Index,cfg.gpu.deviceIndex);
    end

    avail = double(g.AvailableMemory);
    if strcmpi(cfg.gpu.precision,'single'), br=4; else, br=8; end
    bc = 2*br;

    mPlus = 2*n+3;
    mInner = 2*n+1;
    Np = double(mPlus)^3;
    Ni = double(mInner)^3;
    NF = size(basis.qModes,1);
    L = basis.numTerms;
    batch = cfg.taylor.largeWindowCenterBatchSize;

    fieldPlus = 3*Np*bc;
    croppedA = 3*Ni*bc;
    croppedU = 3*Ni*bc;
    massAction = 3*Ni*bc;
    realWorkspace = 7*Ni*br; % material / a,b / LWRQ convolution buffers
    reconBatch = 2*double(NF)*double(batch)*bc + double(L)*double(part.numCenters)*bc;
    estimatedPeakIncrement = fieldPlus + croppedA + croppedU + massAction + realWorkspace + reconBatch;
    safety = cfg.gpu.largeWindowSafetyGiB*2^30;
    required = estimatedPeakIncrement + safety;

    info = struct();
    info.availableBytes = avail;
    info.estimatedPeakIncrementBytes = estimatedPeakIncrement;
    info.safetyBytes = safety;
    info.requiredBytes = required;
    info.plusGrid = mPlus;
    info.innerGrid = mInner;
    info.numCenters = part.numCenters;
    info.blocksPerDimension = part.blocksPerDimension;
    info.phaseRadiusBound = part.rhoBound;
    if isprop(g,'Index'), info.activeDeviceIndex=g.Index; else, info.activeDeviceIndex=cfg.gpu.deviceIndex; end

    fprintf('\n[Large-window GPU preflight]\n');
    fprintf('  window half-width       : %.3f\n',n*h);
    fprintf('  plus / inner grid       : %d^3 / %d^3\n',mPlus,mInner);
    fprintf('  Taylor centers          : %d (%d^3), rho<=%.4f\n', ...
        part.numCenters,part.blocksPerDimension,part.rhoBound);
    fprintf('  current available GPU   : %.2f GiB\n',avail/2^30);
    fprintf('  estimated added peak    : %.2f GiB\n',estimatedPeakIncrement/2^30);
    fprintf('  configured safety margin: %.2f GiB\n',safety/2^30);

    if avail < required
        error(['Insufficient GPU memory for the configured L=4 window study. ', ...
            'Available %.2f GiB, estimated requirement including safety %.2f GiB.'], ...
            avail/2^30,required/2^30);
    end

    % Catch any future accidental device reset introduced inside this routine.
    assert_gpu_basis_alive_exp3(basis,'Large-window preflight (exit)');
    fprintf('  resident Taylor basis   : VALID\n');
    fprintf('  preflight status        : PASS\n');
end
