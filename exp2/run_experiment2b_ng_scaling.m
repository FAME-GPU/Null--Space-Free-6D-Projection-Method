function results = run_experiment2b_ng_scaling(cfg,rootDir)
%RUN_EXPERIMENT2B_NG_SCALING Figure 2(b): reconstruction scaling with N_G.
%
% Key design guarantees:
%   1. The N=5 Fourier eigenproblem is solved only once.
%   2. h is fixed and n is swept through cfg.scaling.nValues.
%   3. Taylor uses p=10 and the smallest k^3 centers satisfying the exact
%      adopted phase-radius threshold from n=40, k=4.
%   4. No full Yee coordinate arrays or reconstructed fields are stored.
%   5. Direct and Taylor have independent 1e4-s budgets.
%   6. e2/eInf are computed only when both methods complete the same N_G.
%   7. A checkpoint is written after every completed scaling size.

    if nargin<2 || isempty(rootDir), rootDir=fileparts(mfilename('fullpath')); end
    outDir = fullfile(rootDir,'output');
    figDir = fullfile(rootDir,'figures');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    if ~exist(figDir,'dir'), mkdir(figDir); end

    validate_config(cfg);
    gpuInfo = gpu_initialize(cfg.gpu);
    if ~cfg.gpu.useGPU
        error('run_experiment2b_ng_scaling:GPURequired','This production scaling run requires a GPU.');
    end

    N = cfg.problem.N;
    NF = (2*N)^6;
    modeIdx = cfg.spectrum.performanceMode;
    nValues = cfg.scaling.nValues(:);
    h = cfg.scaling.h;

    fprintf('\n============================================================\n');
    fprintf('Experiment 2(b): N_G scaling of 3-D Yee reconstruction\n');
    fprintf('N=%d, NF=%d, representative positive mode=%d\n',N,NF,modeIdx);
    fprintf('q=(%.3f, %.3f, %.3f)^T; fixed h=%.5f\n', ...
        cfg.problem.q(1),cfg.problem.q(2),cfg.problem.q(3),h);
    fprintf('n sweep: '); fprintf('%d ',nValues); fprintf('\n');
    fprintf('Independent per-method time budget: %.1f s\n',cfg.scaling.timeoutSeconds);
    fprintf('Memory policy: streamed Yee blocks; no global N_G coordinate/field arrays.\n');
    fprintf('============================================================\n');

    %% ---------------------------------------------------------
    % Solve the Fourier eigenproblem once.
    % ----------------------------------------------------------
    fprintf('\n[1/5] Building the fixed N=5 Fourier system...\n');
    check_gpu_memory_exp2b(cfg,'before Fourier system build');
    wait_for_gpu(); tBuild=tic;
    sys = build_experiment3_system(cfg,N,gpuInfo);
    wait_for_gpu(); systemBuildSeconds=toc(tBuild);

    fprintf('[2/5] Solving the first ten positive eigenmodes once...\n');
    sol = solve_modes_exp3(sys,cfg,cfg.solver.tolCG,cfg.solver.tolLanczos, ...
        cfg.solver.rngSeed+N);
    maxRes = gather_scalar(max(sol.resGEVP(cfg.spectrum.modes)));
    if maxRes > cfg.solver.requiredOriginalGEVPResidual
        warning('run_experiment2b_ng_scaling:GEVPResidual', ...
            'Maximum GEVP residual %.3e exceeds requested %.1e.', ...
            maxRes,cfg.solver.requiredOriginalGEVPResidual);
    end
    if modeIdx<1 || modeIdx>size(sol.X,2)
        error('run_experiment2b_ng_scaling:ModeMissing','Requested mode %d is unavailable.',modeIdx);
    end

    fprintf('[3/5] Extracting reusable mode and projected q-modes...\n');
    XMode = sol.X(:,modeIdx);
    lambdaMode = gather_scalar(sol.lambda(modeIdx));
    qModes = build_projected_q_modes(sys.P,sys.kLift,sys.fourier.Xi,sys.NF,cfg.problem.dim);
    wait_for_gpu(qModes);
    if ~isa(XMode,'gpuArray') || ~isa(qModes,'gpuArray')
        error('run_experiment2b_ng_scaling:GPUResidency','Mode coefficients and qModes must remain on GPU.');
    end

    eigensolverSeconds = sol.wallTimeSeconds;
    outerSteps = sol.outerSteps;
    clear sol sys
    wait_for_gpu();
    memAfterSolve = check_gpu_memory_exp2b(cfg,'after clearing eigensolver arrays');

    %% ---------------------------------------------------------
    % Load optional checkpoint metadata before fixing batch sizes.
    % ----------------------------------------------------------
    checkpointPath = fullfile(outDir,cfg.output.checkpointFile);
    haveCheckpoint = false;
    stateLoaded = struct();
    if cfg.run.resume && exist(checkpointPath,'file')==2
        tmp = load(checkpointPath,'state');
        if isfield(tmp,'state')
            stateLoaded = tmp.state;
            haveCheckpoint = true;
        end
    end

    %% ---------------------------------------------------------
    % Memory-safe Taylor preprocessing and direct block-size choice.
    % ----------------------------------------------------------
    if haveCheckpoint && isfield(stateLoaded,'termBatchSize')
        termBatchSize = stateLoaded.termBatchSize;
    else
        tMem = choose_safe_taylor_batch_exp2b(NF,cfg.scaling.termBatchSizeRequested,qModes(1),cfg);
        termBatchSize = tMem.termBatchSize;
    end

    fprintf('\n[4/5] Precomputing reusable Taylor q-powers (p=%d, term batch=%d)...\n', ...
        cfg.scaling.taylorOrder,termBatchSize);
    pre = precompute_taylor_exp2b(qModes,cfg.scaling.taylorOrder,termBatchSize);
    wait_for_gpu();

    directMem = choose_safe_direct_block_size_exp2b(NF,qModes(1),cfg);
    if haveCheckpoint && isfield(stateLoaded,'directBlockSize')
        directBlockSize = stateLoaded.directBlockSize;
        if directBlockSize > directMem.blockSize
            error('run_experiment2b_ng_scaling:ResumeMemoryMismatch', ...
                ['Checkpoint used direct block size %d, but current live GPU memory safely ', ...
                 'supports only %d. Free GPU memory or start a fresh run.'], ...
                directBlockSize,directMem.blockSize);
        end
    else
        directBlockSize = directMem.blockSize;
    end
    fprintf('  safe direct point batch = %d (raw phase workspace %.2f GiB)\n', ...
        directBlockSize,24*double(NF)*double(directBlockSize)/2^30);
    devNow = gpuDevice;
    fprintf('  live GPU available memory after Taylor preprocessing = %.2f GiB\n', ...
        double(devNow.AvailableMemory)/2^30);

    qL1Max = pre.qL1Max;
    rhoTarget = baseline_phase_radius_exp2b( ...
        cfg.scaling.baselineN,h,cfg.scaling.baselineBlocksPerDimension,qL1Max);
    fprintf('  exact adopted radius threshold = %.12g (displayed in panel (a) as %.2f)\n', ...
        rhoTarget,rhoTarget);

    % Build the full planned geometry table before any expensive scaling run.
    nRows = numel(nValues);
    Ng = zeros(nRows,1);
    kVec = zeros(nRows,1);
    centersVec = zeros(nRows,1);
    rhoVec = zeros(nRows,1);
    maxBlockPtsVec = zeros(nRows,1);
    partCache = cell(nRows,1);
    for i=1:nRows
        nNow = nValues(i);
        choose = choose_partition_for_radius_exp2b(nNow,h,qL1Max,rhoTarget, ...
            cfg.scaling.radiusRelativeTolerance);
        part = build_block_ranges_exp2b(nNow,choose.blocksPerDimension);
        Ng(i) = part.numPoints;
        kVec(i) = choose.blocksPerDimension;
        centersVec(i) = choose.numCenters;
        rhoVec(i) = choose.rhoMax;
        maxBlockPtsVec(i) = part.maxBlockPoints;
        partCache{i} = part;
        if rhoVec(i) > rhoTarget*(1+cfg.scaling.radiusRelativeTolerance)
            error('run_experiment2b_ng_scaling:RadiusViolation','Planned radius exceeds target at n=%d.',nNow);
        end
    end
    idxBase = find(nValues==cfg.scaling.baselineN,1);
    if isempty(idxBase) || kVec(idxBase)~=cfg.scaling.baselineBlocksPerDimension
        error('run_experiment2b_ng_scaling:BaselineMismatch', ...
            'The baseline n=%d must reproduce k=%d.', ...
            cfg.scaling.baselineN,cfg.scaling.baselineBlocksPerDimension);
    end

    fprintf('\n  Planned radius-controlled Taylor partitions:\n');
    fprintf('      n          N_G      k       centers     r_ph,max\n');
    for i=1:nRows
        fprintf('    %3d   %10d     %2d      %6d      %.6f\n', ...
            nValues(i),Ng(i),kVec(i),centersVec(i),rhoVec(i));
    end

    %% ---------------------------------------------------------
    % Initialize or restore the compact scaling table.
    % ----------------------------------------------------------
    scaling = initialize_scaling_table(nValues,Ng,kVec,centersVec,rhoVec, ...
        maxBlockPtsVec,rhoTarget,directBlockSize,termBatchSize,cfg);
    directActive = true;
    taylorActive = true;
    startRow = 1;

    if haveCheckpoint
        validate_checkpoint(stateLoaded,cfg,nValues,rhoTarget,directBlockSize,termBatchSize);
        scaling = stateLoaded.scaling;
        directActive = stateLoaded.directActive;
        taylorActive = stateLoaded.taylorActive;
        done = find(scaling.RowComplete,1,'last');
        if ~isempty(done), startRow=done+1; end
        fprintf('\n[Resume] Loaded checkpoint through row %d/%d. directActive=%d, TaylorActive=%d.\n', ...
            startRow-1,nRows,directActive,taylorActive);
    end

    %% ---------------------------------------------------------
    % Optional one-block warm-up (never counted in plotted times).
    % ----------------------------------------------------------
    if cfg.scaling.warmupKernels && startRow<=nRows && (directActive || taylorActive)
        fprintf('\n[Warm-up] One streamed local block, excluded from timings...\n');
        iw = max(1,min(nRows,startRow));
        pW = partCache{iw};
        [R,rc] = build_yee_block_points_exp2b(pW.ranges(1,:),h,1,qModes(1));
        wait_for_gpu();
        coeffW = XMode(1:NF);
        if taylorActive
            [uW,~] = reconstruct_taylor_group_exp2b(coeffW,qModes,{R},{rc},pre); clear uW
        end
        if directActive
            [uW,~] = reconstruct_direct_group_exp2b(coeffW,qModes,{R},directBlockSize); clear uW
        end
        clear R rc coeffW
        wait_for_gpu();
    end

    %% ---------------------------------------------------------
    % Main N_G sweep.
    % ----------------------------------------------------------
    fprintf('\n[5/5] Running N_G scaling sweep...\n');
    for i=startRow:nRows
        nNow = nValues(i);
        if ~directActive && ~taylorActive
            fprintf('\nBoth methods have exhausted their independent budgets; remaining sizes are skipped.\n');
            scaling.DirectStatus(i:end) = "skipped_after_budget";
            scaling.TaylorStatus(i:end) = "skipped_after_budget";
            break;
        end

        fprintf('\n------------------------------------------------------------\n');
        fprintf('n=%d, N_G=%d, k=%d, centers=%d, r_ph,max=%.6f\n', ...
            nNow,Ng(i),kVec(i),centersVec(i),rhoVec(i));
        fprintf('  direct active=%d; Taylor active=%d\n',directActive,taylorActive);
        check_gpu_memory_exp2b(cfg,sprintf('before n=%d',nNow));

        % Re-check live memory before every scale.  The streamed design makes
        % N_G-independent batches normally stay unchanged, but if MATLAB/GPU
        % memory pressure has increased we reduce a batch proactively instead
        % of risking an out-of-memory failure.  We never increase a batch
        % again within the same run, preserving a conservative timing policy.
        if taylorActive
            tSafeNow=choose_safe_taylor_batch_exp2b(NF,termBatchSize,qModes(1),cfg);
            if tSafeNow.termBatchSize < termBatchSize
                fprintf('  memory guard: Taylor term batch %d -> %d\n',termBatchSize,tSafeNow.termBatchSize);
                termBatchSize=tSafeNow.termBatchSize;
                pre.termBatchSize=termBatchSize;
            end
        end
        if directActive
            dSafeNow=choose_safe_direct_block_size_exp2b(NF,qModes(1),cfg);
            if dSafeNow.blockSize < directBlockSize
                fprintf('  memory guard: direct point batch %d -> %d\n',directBlockSize,dSafeNow.blockSize);
                directBlockSize=dSafeNow.blockSize;
            end
        end
        scaling.TaylorTermBatchSize(i)=termBatchSize;
        scaling.DirectBlockSize(i)=directBlockSize;

        out = run_one_ng_scaling_size_exp2b(XMode,qModes,nNow,h,partCache{i},pre, ...
            directActive,taylorActive,directBlockSize,cfg);

        scaling.DirectTimeSeconds(i) = out.directTimeSeconds;
        scaling.DirectElapsedAtStopSeconds(i) = out.directElapsedAtStopSeconds;
        scaling.TaylorTimeSeconds(i) = out.taylorTimeSeconds;
        scaling.TaylorElapsedAtStopSeconds(i) = out.taylorElapsedAtStopSeconds;
        scaling.RelativeFieldError2(i) = out.relativeError2;
        scaling.RelativeFieldErrorInf(i) = out.relativeErrorInf;
        scaling.DirectStatus(i) = string(out.directStatus);
        scaling.TaylorStatus(i) = string(out.taylorStatus);
        scaling.DirectCompleted(i) = out.directCompleted;
        scaling.TaylorCompleted(i) = out.taylorCompleted;
        scaling.DirectBudgetExceeded(i) = out.directBudgetExceeded;
        scaling.TaylorBudgetExceeded(i) = out.taylorBudgetExceeded;
        scaling.RowComplete(i) = true;

        fprintf('  RESULT direct: %-22s  time=%s\n',out.directStatus,format_time(out.directTimeSeconds,out.directElapsedAtStopSeconds));
        fprintf('  RESULT Taylor: %-22s  time=%s\n',out.taylorStatus,format_time(out.taylorTimeSeconds,out.taylorElapsedAtStopSeconds));
        if isfinite(out.relativeError2)
            fprintf('  errors: e2=%.3e, eInf=%.3e\n',out.relativeError2,out.relativeErrorInf);
        else
            fprintf('  errors: not evaluated (no complete direct/Taylor pair).\n');
        end

        % Independent future states.
        if out.directBudgetExceeded || ~out.directCompleted
            directActive = false;
        end
        if out.taylorBudgetExceeded || ~out.taylorCompleted
            taylorActive = false;
        end

        % Checkpoint after every completed scaling row.
        state = make_checkpoint_state(scaling,directActive,taylorActive,cfg,nValues, ...
            rhoTarget,directBlockSize,termBatchSize);
        save(checkpointPath,'state','-v7.3');
        writetable(scaling,fullfile(outDir,cfg.output.scalingCSV));
    end

    %% Final validation and output.
    results = struct();
    results.config = cfg;
    results.gpu = gpuInfo;
    results.scaling = scaling;
    results.rhoTarget = rhoTarget;
    results.qL1Max = qL1Max;
    results.mode = modeIdx;
    results.lambdaMode = lambdaMode;
    results.N = N;
    results.NF = NF;
    results.directBlockSize = directBlockSize;
    results.termBatchSize = termBatchSize;
    results.timeoutSeconds = cfg.scaling.timeoutSeconds;
    results.eigensolver = struct('seconds',eigensolverSeconds,'outerSteps',outerSteps, ...
        'maxOriginalGEVPResidual',maxRes,'systemBuildSeconds',systemBuildSeconds);
    results.memoryAfterSolve = memAfterSolve;

    check_experiment2b_results(results,cfg,true);
    save(fullfile(outDir,cfg.output.resultFile),'results','-v7.3');
    writetable(scaling,fullfile(outDir,cfg.output.scalingCSV));
    write_experiment2b_summary(results,cfg,rootDir);
    if cfg.run.makePlot
        plot_experiment2b_ng_scaling(results,cfg,rootDir);
    end

    state = make_checkpoint_state(scaling,directActive,taylorActive,cfg,nValues, ...
        rhoTarget,directBlockSize,termBatchSize);
    state.finished = all(scaling.RowComplete | ...
        (scaling.DirectStatus=="skipped_after_budget" & scaling.TaylorStatus=="skipped_after_budget"));
    save(checkpointPath,'state','-v7.3');

    clear XMode qModes pre partCache
    wait_for_gpu();
    fprintf('\nExperiment 2(b) scaling outputs written to %s\n',outDir);
end

function validate_config(cfg)
    if cfg.problem.N~=5
        error('validate_config:BadN','This package is deliberately fixed at N=5.');
    end
    if cfg.spectrum.performanceMode~=10 || cfg.spectrum.numEig<10
        error('validate_config:BadMode','The panel-(b) scaling package requires positive mode 10.');
    end
    n = cfg.scaling.nValues(:);
    if isempty(n) || any(n<=0) || any(diff(n)<=0) || any(n~=round(n))
        error('validate_config:BadNValues','nValues must be strictly increasing positive integers.');
    end
    if ~any(n==cfg.scaling.baselineN)
        error('validate_config:MissingBaseline','nValues must contain the adopted baseline n.');
    end
    if cfg.scaling.h<=0 || cfg.scaling.timeoutSeconds<=0
        error('validate_config:BadPositiveParameter','h and timeoutSeconds must be positive.');
    end
    if cfg.scaling.centersPerTimeoutCheck<1
        error('validate_config:BadTimeoutChunk','centersPerTimeoutCheck must be >=1.');
    end
end

function T = initialize_scaling_table(nValues,Ng,kVec,centers,rho,maxPts,rhoTarget,directBlock,termBatch,cfg)
    m=numel(nValues);
    T=table(nValues,Ng,kVec,centers,rho,maxPts, ...
        repmat(rhoTarget,m,1),repmat(cfg.scaling.taylorOrder,m,1), ...
        repmat(termBatch,m,1),repmat(directBlock,m,1), ...
        NaN(m,1),NaN(m,1),NaN(m,1),NaN(m,1),NaN(m,1),NaN(m,1), ...
        repmat("pending",m,1),repmat("pending",m,1), ...
        false(m,1),false(m,1),false(m,1),false(m,1),false(m,1), ...
        'VariableNames',{'n','Ng','BlocksPerDimension','NumCenters','RhoMax','MaxBlockPoints', ...
        'RhoTarget','TaylorOrder','TaylorTermBatchSize','DirectBlockSize', ...
        'DirectTimeSeconds','DirectElapsedAtStopSeconds','TaylorTimeSeconds','TaylorElapsedAtStopSeconds', ...
        'RelativeFieldError2','RelativeFieldErrorInf','DirectStatus','TaylorStatus', ...
        'DirectCompleted','TaylorCompleted','DirectBudgetExceeded','TaylorBudgetExceeded','RowComplete'});
end

function state = make_checkpoint_state(scaling,directActive,taylorActive,cfg,nValues,rhoTarget,directBlock,termBatch)
    state=struct();
    state.version=1;
    state.scaling=scaling;
    state.directActive=logical(directActive);
    state.taylorActive=logical(taylorActive);
    state.nValues=nValues(:);
    state.rhoTarget=rhoTarget;
    state.directBlockSize=directBlock;
    state.termBatchSize=termBatch;
    state.timeoutSeconds=cfg.scaling.timeoutSeconds;
    state.taylorOrder=cfg.scaling.taylorOrder;
    state.h=cfg.scaling.h;
    state.finished=false;
end

function validate_checkpoint(state,cfg,nValues,rhoTarget,directBlock,termBatch)
    req={'scaling','directActive','taylorActive','nValues','rhoTarget','directBlockSize','termBatchSize','timeoutSeconds','taylorOrder','h'};
    for j=1:numel(req)
        if ~isfield(state,req{j})
            error('validate_checkpoint:MissingField','Checkpoint lacks field %s.',req{j});
        end
    end
    if ~isequal(state.nValues(:),nValues(:)) || state.taylorOrder~=cfg.scaling.taylorOrder || ...
            state.h~=cfg.scaling.h || state.timeoutSeconds~=cfg.scaling.timeoutSeconds
        error('validate_checkpoint:ConfigMismatch','Checkpoint configuration does not match the current run.');
    end
    if abs(state.rhoTarget-rhoTarget)>1e-11*max(1,abs(rhoTarget))
        error('validate_checkpoint:RadiusMismatch','Checkpoint radius threshold differs from current qModes.');
    end
    if state.directBlockSize~=directBlock || state.termBatchSize~=termBatch
        error('validate_checkpoint:BatchMismatch','Checkpoint batch sizes do not match current batch sizes.');
    end
end

function s=format_time(fullTime,elapsed)
    if isfinite(fullTime)
        s=sprintf('%.4f s',fullTime);
    else
        s=sprintf('incomplete; %.4f s accumulated',elapsed);
    end
end
