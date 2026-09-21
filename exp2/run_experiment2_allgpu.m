function results = run_experiment2_allgpu(cfg,rootDir)
%RUN_EXPERIMENT2_ALLGPU N=5 all-GPU Yee-field reconstruction experiment.
%
% All large numerical arrays remain gpuArray from eigensolve through field
% reconstruction and cropped-Yee diagnostics.  Only scalar diagnostics,
% compact tables, block-index metadata, saving, and plotting use the CPU.

    if nargin<2 || isempty(rootDir), rootDir=fileparts(mfilename('fullpath')); end
    outDir=fullfile(rootDir,'output');
    figDir=fullfile(rootDir,'figures');
    if ~exist(outDir,'dir'), mkdir(outDir); end
    if ~exist(figDir,'dir'), mkdir(figDir); end

    gpuInfo=gpu_initialize(cfg.gpu);
    if ~cfg.gpu.useGPU, error('Final Experiment 2 requires GPU execution.'); end
    N=cfg.problem.N; modes=cfg.spectrum.modes(:).';
    if N~=5, error('This production package is configured for N=5.'); end
    if numel(modes)~=10 || any(modes~=(1:10)), error('Expected modes 1:10.'); end
    modeTrade=cfg.tradeoff.mode; modeWorkflow=cfg.workflow.mode;
    if modeTrade~=modeWorkflow, error('Trade-off and workflow must use the same mode.'); end
    jMode=find(modes==modeTrade,1);
    if isempty(jMode), error('Representative mode must belong to 1:10.'); end
    if abs(cfg.yee.n*cfg.yee.h-0.5)>100*eps, error('Required n*h=0.5.'); end

    fprintf('\n============================================================\n');
    fprintf('Experiment 2: all-GPU 3-D Yee-field reconstruction\n');
    fprintf('N=%d, NF=%d, reduced dim=%d, mode=%d\n',N,(2*N)^6,2*(2*N)^6,modeTrade);
    fprintf('q=(%.3f, %.3f, %.3f)^T; Yee n=%d, h=%.5f\n', ...
        cfg.problem.q(1),cfg.problem.q(2),cfg.problem.q(3),cfg.yee.n,cfg.yee.h);
    fprintf('Direct block=%d; adopted Taylor p=%d, centers=%d\n', ...
        cfg.direct.blockSize,cfg.taylor.performanceOrder,cfg.taylor.blocksPerDimension^3);
    fprintf('All large reconstruction arrays remain on GPU.\n');
    fprintf('============================================================\n');

    if ~cfg.run.solve, error('Production run requires cfg.run.solve=true.'); end
    optDirect=struct('blockSize',cfg.direct.blockSize);
    pPerf=cfg.taylor.performanceOrder;

    %% System + eigensolver (GPU)
    fprintf('\n[System] Building N=5 exact scalar rank-1 Fourier system...\n');
    wait_for_gpu(); tBuild=tic;
    sys=build_experiment3_system(cfg,N,gpuInfo);
    wait_for_gpu(); systemBuildSeconds=toc(tBuild);

    fprintf('[Eigensolver] Computing first ten positive modes on GPU...\n');
    sol=solve_modes_exp3(sys,cfg,cfg.solver.tolCG,cfg.solver.tolLanczos,cfg.solver.rngSeed+N);
    maxRes=max(sol.resGEVP(modes));
    fprintf('  max original-GEVP residual = %.3e\n',maxRes);
    if maxRes>cfg.solver.requiredOriginalGEVPResidual
        warning('Maximum GEVP residual %.3e exceeds requested %.1e.',maxRes,cfg.solver.requiredOriginalGEVPResidual);
    end
    if ~isa(sol.X,'gpuArray'), error('Eigenvectors unexpectedly left GPU.'); end

    %% GPU physical-space preparation
    wait_for_gpu(); tQ=tic;
    qModes=build_projected_q_modes(sys.P,sys.kLift,sys.fourier.Xi,sys.NF,cfg.problem.dim);
    wait_for_gpu(qModes); qModesBuildSeconds=toc(tQ);
    if ~isa(qModes,'gpuArray'), error('qModes must remain on GPU.'); end

    wait_for_gpu(); tYee=tic;
    yee=build_yee_points_exp3(cfg.yee.n,cfg.yee.h,sys.P(1));
    wait_for_gpu(yee.r3); yeeGridBuildSeconds=toc(tYee);
    if ~isa(yee.r1,'gpuArray') || ~isa(yee.r2,'gpuArray') || ~isa(yee.r3,'gpuArray')
        error('Yee points must remain on GPU.');
    end

    XModeGPU=sol.X(:,jMode);
    if ~isa(XModeGPU,'gpuArray'), error('Representative Fourier eigenvector is not on GPU.'); end

    % Block lists are small integer metadata; numerical reconstruction stays GPU.
    tBlocks=tic;
    blocksAdopt=build_yee_block_partition_exp3(yee.szPlus,cfg.taylor.blocksPerDimension);
    taylorBlockPartitionSeconds=toc(tBlocks);

    commonPreparationSeconds=systemBuildSeconds+qModesBuildSeconds+yeeGridBuildSeconds;
    directOtherPreparationSeconds=commonPreparationSeconds;
    taylorOtherPreparationSeconds=commonPreparationSeconds+taylorBlockPartitionSeconds;

    %% GPU blockwise direct reference
    fprintf('\n[Reference] GPU blockwise direct reconstruction of mode %d...\n',modeTrade);
    if cfg.direct.warmupReference
        [fw,~]=reconstruct_three_components_exp3(XModeGPU,qModes,yee,'direct',optDirect); clear fw
    end
    [refField,infoRef]=reconstruct_three_components_exp3(XModeGPU,qModes,yee,'direct',optDirect);
    assert_gpu_field(refField,'direct reference');
    refA=apply_cropped_A_exp3(refField,cfg.yee.h);
    assert_gpu_field(refA,'cropped direct A-field');
    directReconstructionSeconds=infoRef.timeSeconds;
    directAuxMiB=infoRef.estimatedPeakAuxBytes/2^20;
    fprintf('  GPU direct reconstruction = %.4f s\n',directReconstructionSeconds);

    %% GPU Taylor trade-off
    pTrade=cfg.tradeoff.orders(:).'; kTrade=cfg.tradeoff.blocksPerDimension(:).';
    nRows=numel(pTrade)*numel(kTrade);
    tradeP=zeros(nRows,1); tradeK=zeros(nRows,1); tradeCenters=zeros(nRows,1);
    tradeRho=zeros(nRows,1); tradeE2=zeros(nRows,1); tradeTime=zeros(nRows,1); tradeAuxMiB=zeros(nRows,1);
    adoptedE2=NaN; adoptedEA=NaN; adoptedRho=NaN; adoptedTime=NaN; adoptedAuxMiB=NaN;

    row=0;
    fprintf('\n[Figure 2(a)] GPU Taylor accuracy-cost trade-off...\n');
    for ip=1:numel(pTrade)
        pNow=pTrade(ip);
        for ik=1:numel(kTrade)
            kNow=kTrade(ik); nCenters=kNow^3;
            if kNow==1
                methodNow='single';
                optNow=struct('order',pNow,'termBatchSize',cfg.taylor.termBatchSize,'center',cfg.taylor.singleCenter);
            else
                if kNow==cfg.taylor.blocksPerDimension
                    blocksNow=blocksAdopt;
                else
                    blocksNow=build_yee_block_partition_exp3(yee.szPlus,kNow);
                end
                methodNow='multi';
                optNow=struct('order',pNow,'termBatchSize',cfg.taylor.termBatchSize,'blocks',{blocksNow});
            end
            if cfg.tradeoff.warmupEachConfiguration
                [fw,~]=reconstruct_three_components_exp3(XModeGPU,qModes,yee,methodNow,optNow); clear fw
            end
            [ft,it]=reconstruct_three_components_exp3(XModeGPU,qModes,yee,methodNow,optNow);
            assert_gpu_field(ft,sprintf('Taylor p=%d centers=%d',pNow,nCenters));
            et=compute_field_errors_exp3(ft,refField);
            row=row+1;
            tradeP(row)=pNow; tradeK(row)=kNow; tradeCenters(row)=nCenters;
            tradeRho(row)=it.rhoMax; tradeE2(row)=et.e2; tradeTime(row)=it.timeSeconds;
            tradeAuxMiB(row)=it.estimatedPeakAuxBytes/2^20;
            if pNow==pPerf && kNow==cfg.taylor.blocksPerDimension
                adoptedE2=et.e2;
                adoptedEA=compute_A_error_exp3(ft,refA,cfg.yee.h);
                adoptedRho=it.rhoMax; adoptedTime=it.timeSeconds;
                adoptedAuxMiB=it.estimatedPeakAuxBytes/2^20;
            end
            fprintf('  p=%2d, centers=%3d: rph=%.6f, e2=%.3e, GPU time=%.4f s\n', ...
                pNow,nCenters,it.rhoMax,et.e2,it.timeSeconds);
            clear ft it
        end
    end

    tradeoff=table(tradeP,tradeK,tradeCenters,tradeRho,tradeE2,tradeTime,tradeAuxMiB, ...
        'VariableNames',{'p','BlocksPerDimension','NumCenters','RhoMax', ...
        'RelativeFieldErrorMode10','GPUTimeSeconds','EstimatedPeakAuxMemoryMiB'});
    writetable(tradeoff,fullfile(outDir,cfg.output.tradeoffCSV));

    rowSingle=(tradeoff.p==pPerf & tradeoff.NumCenters==1);
    rowAdopt=(tradeoff.p==pPerf & tradeoff.NumCenters==cfg.taylor.blocksPerDimension^3);
    rphSingle=tradeoff.RhoMax(find(rowSingle,1)); rphAdopt=tradeoff.RhoMax(find(rowAdopt,1));
    singleE2=tradeoff.RelativeFieldErrorMode10(find(rowSingle,1));
    singleTime=tradeoff.GPUTimeSeconds(find(rowSingle,1));
    singleAuxMiB=tradeoff.EstimatedPeakAuxMemoryMiB(find(rowSingle,1));
    if ~isfinite(adoptedTime), error('Adopted p=10/64-center configuration missing.'); end

    %% Complete GPU workflow timing
    eigensolverSeconds=sol.wallTimeSeconds;
    % Manuscript timing uses only the large numerical GPU stages, so every
    % displayed timing has the same GPU execution meaning.  Small host-side
    % metadata/setup is stored separately and is not folded into the plotted
    % GPU workflow total.
    gpuTotalDirect=eigensolverSeconds+directReconstructionSeconds;
    gpuTotalTaylor=eigensolverSeconds+adoptedTime;
    fullTotalDirect=gpuTotalDirect+directOtherPreparationSeconds;
    fullTotalTaylor=gpuTotalTaylor+taylorOtherPreparationSeconds;
    workflowRatio=gpuTotalDirect/gpuTotalTaylor;
    workflow=table({'Blockwise direct';'Multi-center Taylor'}, ...
        [eigensolverSeconds;eigensolverSeconds], ...
        [directReconstructionSeconds;adoptedTime], ...
        [gpuTotalDirect;gpuTotalTaylor], ...
        [directOtherPreparationSeconds;taylorOtherPreparationSeconds], ...
        [fullTotalDirect;fullTotalTaylor], ...
        'VariableNames',{'Workflow','EigensolverSeconds','ReconstructionSeconds', ...
        'GPUWorkflowTotalSeconds','OtherPreparationSeconds','FullWorkflowTotalSeconds'});
    writetable(workflow,fullfile(outDir,cfg.output.workflowCSV));

    fprintf('\n[Figure 2(b)] GPU workflow timing:\n');
    fprintf('  eigensolver common       = %.4f s\n',eigensolverSeconds);
    fprintf('  direct reconstruction    = %.4f s\n',directReconstructionSeconds);
    fprintf('  Taylor reconstruction    = %.4f s\n',adoptedTime);
    fprintf('  GPU numerical total      = %.4f / %.4f s (%.3fx)\n',gpuTotalDirect,gpuTotalTaylor,workflowRatio);
    fprintf('  host/setup-inclusive total= %.4f / %.4f s\n',fullTotalDirect,fullTotalTaylor);

    %% Diagnostic tables
    cg=sol.cgStats; et=sol.timing;
    eigensolverBreakdown=table(N,sys.NF,sys.nr,sol.outerSteps, ...
        et.totalEigensolverSeconds,et.lanczosCoreSeconds,et.innerMhatCGSeconds, ...
        et.inverseActionOtherSeconds,et.outerLanczosAlgebraSeconds, ...
        et.eigenvectorRecoverySeconds,et.fourierDiagnosticsSeconds, ...
        cg.calls,cg.totalIterations,cg.averageIterations,cg.maxRelativeResidual, ...
        'VariableNames',{'N','NF','ReducedDimension','OuterLanczosSteps', ...
        'TotalEigensolverSeconds','LanczosCoreSeconds','InnerMhatCGSeconds', ...
        'InverseActionOtherSeconds','OuterLanczosAlgebraSeconds','EigenvectorRecoverySeconds', ...
        'FourierDiagnosticsSeconds','InnerCGCalls','InnerCGTotalIterations', ...
        'InnerCGAverageIterations','InnerCGMaxRelativeResidual'});
    writetable(eigensolverBreakdown,fullfile(outDir,cfg.output.eigensolverBreakdownCSV));

    prepStage={'System construction';'Projected q-mode construction';'GPU Yee-grid construction'; ...
        'Taylor block-index metadata';'Direct other preparation total';'Taylor other preparation total'};
    prepSeconds=[systemBuildSeconds;qModesBuildSeconds;yeeGridBuildSeconds;taylorBlockPartitionSeconds; ...
        directOtherPreparationSeconds;taylorOtherPreparationSeconds];
    preparationBreakdown=table(prepStage,prepSeconds,'VariableNames',{'Stage','Seconds'});
    writetable(preparationBreakdown,fullfile(outDir,cfg.output.preparationBreakdownCSV));

    performance=table([N;N;N],{'Blockwise direct';'Single-center Taylor';'Multi-center Taylor'}, ...
        [cfg.direct.blockSize;1;cfg.taylor.blocksPerDimension^3], ...
        [directReconstructionSeconds;singleTime;adoptedTime], ...
        [directAuxMiB;singleAuxMiB;adoptedAuxMiB],[NaN;singleE2;adoptedE2], ...
        'VariableNames',{'N','Method','BatchSizeOrCenters','GPUReconstructionTimeSeconds', ...
        'EstimatedPeakAuxMemoryMiB','RelativeFieldError'});
    writetable(performance,fullfile(outDir,cfg.output.performanceCSV));

    Ng=(2*cfg.yee.n+3)^3;
    denseGiB=double(Ng)*double((2*N)^6)*16/2^30;
    results=struct(); results.config=cfg; results.gpu=gpuInfo; results.tradeoff=tradeoff;
    results.workflow=workflow; results.workflowRatio=workflowRatio;
    results.eigensolverBreakdown=eigensolverBreakdown; results.preparationBreakdown=preparationBreakdown;
    results.performance=performance;
    results.adopted=struct('order',pPerf,'numCenters',cfg.taylor.blocksPerDimension^3, ...
        'rhoMax',adoptedRho,'rhoSingle',rphSingle,'rhoAdopted',rphAdopt, ...
        'mode',modeTrade,'relativeFieldError',adoptedE2,'curlCurlActionError',adoptedEA, ...
        'reconstructionSeconds',adoptedTime,'estimatedPeakAuxMemoryMiB',adoptedAuxMiB);
    results.N5=struct('N',N,'NF',sys.NF,'reducedDimension',sys.nr,'lambda',to_cpu(sol.lambda), ...
        'outerSteps',sol.outerSteps,'maxOriginalGEVPResidual',maxRes, ...
        'totalInnerCGIterations',cg.totalIterations,'eigensolverTimeSeconds',sol.wallTimeSeconds, ...
        'systemBuildTimeSeconds',systemBuildSeconds,'rank1M',sys.mass.info.M,'rank1MOverNF',sys.mass.info.MOverNF);
    results.reference=struct('Ng',Ng,'n',cfg.yee.n,'h',cfg.yee.h, ...
        'windowWidth',2*cfg.yee.n*cfg.yee.h,'densePhasePerComponentGiB',denseGiB, ...
        'directReconstructionSeconds',directReconstructionSeconds, ...
        'directEstimatedPeakAuxMemoryMiB',directAuxMiB);
    results.notes=struct( ...
        'figure2a','N=5 mode 10, GPU: p={6,8,10}, centers=1^3,...,6^3.', ...
        'workflowDiagnostic','Legacy workflow timing retained as a diagnostic only; final Fig. 2(b) is generated by the N_G scaling experiment.', ...
        'gpuPolicy','Eigensolve, reconstruction, cropped Yee action, and vector norms stay on GPU.', ...
        'metadata','Small block-index metadata is generated on CPU and is reported separately.');

    save(fullfile(outDir,cfg.output.resultFile),'results','-v7.3');
    write_experiment2_allgpu_summary(results,cfg,rootDir);
    if cfg.run.makePlots, plot_experiment2a_tradeoff(results,cfg,rootDir); end
    clear refField refA sys sol qModes yee XModeGPU
    fprintf('\nOutputs written to %s\n',outDir);
end

function assert_gpu_field(field,label)
    if ~isa(field.u1,'gpuArray') || ~isa(field.u2,'gpuArray') || ~isa(field.u3,'gpuArray')
        error('%s unexpectedly left GPU.',label);
    end
end
