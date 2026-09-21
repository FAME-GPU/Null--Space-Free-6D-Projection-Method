function results = run_experiment4(cfg)
%RUN_EXPERIMENT4 Final all-GPU N=5 physical closure/LWRQ validation.
%
% Fig. 3(a): fixed-half-width mesh refinement plus an h=0.025 window study
% extending to half-width L=4.
% Fig. 3(b): relative local LWRQ deviation on z=0 and x=0 at L=4.

    if nargin<1 || isempty(cfg), cfg=make_config_experiment4(); end
    root=setup_experiment4();
    totalTimer=tic;

    fprintf('\n============================================================\n');
    fprintf('Experiment 3: all-GPU N=5 tracked-mode physical validation\n');
    fprintf('Tracked modes=[%d,%d], q=[%.5f %.5f %.5f]^T\n', ...
        cfg.tracking.n5Modes(1),cfg.tracking.n5Modes(2),cfg.problem.q);
    fprintf('Taylor p=%d, rho target=%.2f; window max half-width=%.3f\n', ...
        cfg.taylor.order,cfg.taylor.phaseRadiusTarget,cfg.window.masterN*cfg.window.h);
    fprintf('============================================================\n');

    [selected,eigenpairSource]=get_selected_eigenpairs_exp4(cfg,root);
    fprintf('[Exp3] eigenvalues: mode %d=%.12e, mode %d=%.12e\n', ...
        selected.modes(1),selected.lambda6D(1),selected.modes(2),selected.lambda6D(2));

    gpu_initialize(cfg.gpu);
    prototype=gpuArray(cast(0,cfg.gpu.precision));
    qModes=gpuArray(cast(selected.qModesCPU,cfg.gpu.precision));

    fprintf('[Exp3] Building reusable GPU Taylor Fourier basis...\n');
    tBasis=tic;
    basis=prepare_taylor_fourier_basis_exp4(qModes,cfg.taylor.order);
    wait_for_gpu(basis.Qalpha); basisTime=toc(tBasis);
    fprintf('  terms=%d, Qalpha=%.2f GiB, build=%.2f s, qL1max=%.6f\n', ...
        basis.numTerms,basis.estimatedQalphaBytes/2^30,basisTime,basis.qL1Max);
    clear qModes

    nm=numel(selected.modes);
    nMesh=numel(cfg.mesh.h);
    meshCases=repmat(empty_case_struct(),nMesh*nm,1);
    meshSummary=repmat(empty_summary_struct(),nMesh,1);
    caseCounter=0;
    streamingValidation=struct('performed',false,'mode',NaN,'h',NaN,'n',NaN, ...
        'relativeFieldDifference',NaN,'tolerance',cfg.run.streamingValidationRelTol);

    %% Fig. 3(a), main panel: mesh refinement at fixed half-width 0.5.
    for ic=1:nMesh
        h=cfg.mesh.h(ic); n=cfg.mesh.n(ic);
        if abs(h*n-0.5)>1e-12, error('Mesh case does not preserve half-width 0.5.'); end
        yee=build_yee_points_exp3(n,h,prototype);
        if cfg.run.requireGPUPhysicalKernels && ~isa(yee.r1,'gpuArray')
            error('Yee coordinates unexpectedly left GPU.');
        end
        part=build_auto_taylor_partition_exp4(yee,basis,cfg);
        fprintf('\n[Mesh %d/%d] h=%g n=%d plus=%d^3 centers=%d (%d^3) rho<=%.4f\n', ...
            ic,nMesh,h,n,yee.szPlus(1),part.numCenters,part.blocksPerDimension,part.rhoBound);
        [fields,reconInfo]=reconstruct_tracked_modes_exp4(selected.Xcpu,basis,yee,part,cfg);
        fprintf('  joint GPU reconstruction: %.3f s\n',reconInfo.timeSeconds);

        % Validate the memory-efficient streaming path once on the default
        % grid before it is trusted for the expensive L=4 reconstruction.
        if cfg.run.validateLargeWindowStreamingAtDefault && ...
                abs(h-cfg.default.h)<1e-14 && n==cfg.default.n
            partLight=build_auto_taylor_partition_light_exp3(n,h,basis,cfg);
            [fStream,streamInfo]=reconstruct_single_mode_large_window_exp3( ...
                selected.Xcpu(:,1),basis,n,h,partLight,cfg);
            relStream=relative_field_difference_exp3(fStream,fields{1});
            streamingValidation=struct('performed',true,'mode',selected.modes(1), ...
                'h',h,'n',n,'relativeFieldDifference',relStream, ...
                'tolerance',cfg.run.streamingValidationRelTol, ...
                'timeSeconds',streamInfo.timeSeconds);
            fprintf('  streaming-path validation (mode %d): rel field diff=%.3e\n', ...
                selected.modes(1),relStream);
            if relStream>cfg.run.streamingValidationRelTol
                error('Large-window streaming reconstruction failed default-grid validation.');
            end
            clear fStream streamInfo partLight
            wait_for_gpu();
        end

        metrics=cell(nm,1);
        for j=1:nm
            [m,~]=compute_yee_lwrq_metrics_exp4(fields{j},selected.lambda6D(j),cfg,h,n,false);
            m.mode=selected.modes(j);
            m.reconstructionTimeSeconds=reconInfo.timeSeconds;
            m.numCenters=part.numCenters;
            m.blocksPerDimension=part.blocksPerDimension;
            m.phaseRadiusBound=part.rhoBound;
            m.study='mesh';
            metrics{j}=m;
            caseCounter=caseCounter+1;
            meshCases(caseCounter)=m;
            fprintf('  mode %3d: eta3D=%.3e nu=%.3e deltaLambda=%.3e identity=%.3e\n', ...
                m.mode,m.eta3D,m.nuLWRQ,m.deltaLambda,m.deltaLWRQIdentity);
        end
        meshSummary(ic)=summarize_case(metrics,h,n,part,reconInfo.timeSeconds,'mesh');
        clear fields yee part metrics reconInfo
        wait_for_gpu();
    end
    meshCases=meshCases(1:caseCounter);

    %% Fig. 3(a) inset + Fig. 3(b) data: one maximum L=4 field per tracked mode.
    hW=cfg.window.h;
    nMaster=cfg.window.masterN;
    if abs(hW*nMaster-4.0)>1e-12
        error('The configured window master must have half-width 4.');
    end
    partW=build_auto_taylor_partition_light_exp3(nMaster,hW,basis,cfg);
    preflight=check_large_window_gpu_capacity_exp3(cfg,basis,nMaster,hW,partW);

    fprintf('\n[Window master] h=%g n=%d half-width=%.3f plus=%d^3 centers=%d (%d^3) rho<=%.4f\n', ...
        hW,nMaster,hW*nMaster,2*nMaster+3,partW.numCenters,partW.blocksPerDimension,partW.rhoBound);

    nWin=numel(cfg.window.n);
    windowMetrics=cell(nWin,nm);
    windowCases=repmat(empty_case_struct(),nWin*nm,1);
    spatialMaxWindow=cell(nm,1);
    reconTimes=zeros(1,nm);
    reconInfoPerMode=cell(nm,1);
    caseCounter=0;

    fprintf('[Window sensitivity] each mode is reconstructed once on L=4, then exactly cropped.\n');
    for j=1:nm
        mode=selected.modes(j);
        fprintf('\n  [mode %d] streaming master reconstruction...\n',mode);
        [masterField,reconInfoW]=reconstruct_single_mode_large_window_exp3( ...
            selected.Xcpu(:,j),basis,nMaster,hW,partW,cfg);
        reconTimes(j)=reconInfoW.timeSeconds;
        reconInfoPerMode{j}=reconInfoW;
        fprintf('    reconstruction time: %.3f s\n',reconInfoW.timeSeconds);

        for iw=1:nWin
            n=cfg.window.n(iw);
            if n>nMaster, error('Window crop n=%d exceeds master n=%d.',n,nMaster); end
            if n==nMaster
                f=masterField;
            else
                f=crop_plus_field_exp4(masterField,nMaster,n);
            end
            if cfg.run.requireGPUPhysicalKernels && ~isa(f.u1,'gpuArray')
                error('Window crop unexpectedly left GPU.');
            end

            saveSpatial=cfg.run.saveMaxWindowSpatialSlices && n==nMaster;
            [m,sp]=compute_yee_lwrq_metrics_exp4(f,selected.lambda6D(j),cfg,hW,n,saveSpatial);
            m.mode=mode;
            m.reconstructionTimeSeconds=0;
            m.numCenters=NaN;
            m.blocksPerDimension=NaN;
            m.phaseRadiusBound=NaN;
            m.study='window';
            windowMetrics{iw,j}=m;
            caseCounter=caseCounter+1;
            windowCases(caseCounter)=m;

            if saveSpatial
                sp.mode=mode;
                spatialMaxWindow{j}=sp;
            end

            fprintf('    L=%.3f n=%3d: eta3D=%.3e nu=%.3e deltaLambda=%.3e identity=%.2e\n', ...
                n*hW,n,m.eta3D,m.nuLWRQ,m.deltaLambda,m.deltaLWRQIdentity);
            if n~=nMaster
                clear f
            end
        end

        clear masterField reconInfoW
        wait_for_gpu();
    end
    windowCases=windowCases(1:caseCounter);

    windowSummary=repmat(empty_summary_struct(),nWin,1);
    for iw=1:nWin
        metrics=windowMetrics(iw,:).';
        windowSummary(iw)=summarize_case(metrics,hW,cfg.window.n(iw),[],0,'window');
    end

    % Strong post-run checks before discarding the large Taylor basis.
    validate_window_results(cfg,windowMetrics,spatialMaxWindow,partW);

    clear windowMetrics basis
    wait_for_gpu();

    results=struct();
    results.cfg=cfg;
    results.eigenpairSource=eigenpairSource;
    results.eigen=rmfield_if_present(selected,{'Xcpu','qModesCPU'});
    results.taylorBasisBuildTimeSeconds=basisTime;
    results.meshCases=meshCases;
    results.meshSummary=meshSummary;
    results.windowCases=windowCases;
    results.windowSummary=windowSummary;
    results.streamingReconstructionValidation=streamingValidation;
    results.largeWindowPreflight=preflight;
    results.windowMasterReconstruction=struct('h',hW,'n',nMaster, ...
        'halfWidth',hW*nMaster,'perModeTimeSeconds',reconTimes, ...
        'totalTimeSeconds',sum(reconTimes),'numCenters',partW.numCenters, ...
        'blocksPerDimension',partW.blocksPerDimension,'phaseRadiusBound',partW.rhoBound, ...
        'method','streaming-large-window-1D-geometry','perModeInfo',{reconInfoPerMode});
    results.spatialMaxWindow=spatialMaxWindow;
    results.totalWallTimeSeconds=toc(totalTimer);
    results.physicalKernels='GPU';

    write_experiment4_outputs(results,root);
    fprintf('\n============================================================\n');
    fprintf('Experiment 3 physical validation COMPLETE: %.2f min\n',results.totalWallTimeSeconds/60);
    fprintf('Results: %s\n',fullfile(root,'output',cfg.output.resultFile));
    fprintf('============================================================\n');
end


function rel=relative_field_difference_exp3(A,B)
    num=sum(abs(A.u1(:)-B.u1(:)).^2) + ...
        sum(abs(A.u2(:)-B.u2(:)).^2) + ...
        sum(abs(A.u3(:)-B.u3(:)).^2);
    den=sum(abs(B.u1(:)).^2) + sum(abs(B.u2(:)).^2) + sum(abs(B.u3(:)).^2);
    rel=gather_scalar(sqrt(num/den));
end

function validate_window_results(cfg,windowMetrics,spatialMaxWindow,partW)
    expectedN=[10 15 20 30 40 60 80 100 120 140 160];
    if ~isequal(cfg.window.n,expectedN)
        error('Window n-list changed unexpectedly.');
    end
    if partW.rhoBound>cfg.taylor.phaseRadiusTarget*(1+cfg.taylor.phaseRadiusSlack)
        error('Large-window Taylor phase-radius bound exceeds target.');
    end
    if any(cellfun(@isempty,windowMetrics(:)))
        error('At least one mode/window diagnostic is missing.');
    end
    for j=1:numel(spatialMaxWindow)
        S=spatialMaxWindow{j};
        if isempty(S), error('Missing maximum-window spatial slices for tracked mode %d.',j); end
        if S.n~=160 || abs(S.halfWidth-4)>1e-12 || abs(S.h-0.025)>1e-14
            error('Spatial slice metadata is not the required h=0.025, n=160, L=4 case.');
        end
        if ~isequal(size(S.relativeDeviation_z0),[321 321]) || ...
           ~isequal(size(S.relativeDeviation_x0),[321 321])
            error('Maximum-window spatial slices must be 321-by-321.');
        end
        if any(~isfinite(S.relativeDeviation_z0(:))) || any(~isfinite(S.relativeDeviation_x0(:)))
            error('Nonfinite value found in maximum-window spatial slices.');
        end
        if abs(S.coord(1)+4)>1e-12 || abs(S.coord(end)-4)>1e-12
            error('Maximum-window spatial coordinate range is not [-4,4].');
        end
    end
end

function s=empty_case_struct()
    s=struct('h',[],'n',[],'halfWidth',[],'lambda6D',[],'lambdaCrop',[], ...
        'lambda3D',[],'eta3D',[],'etaUsing6DLambda',[],'deltaLambda',[], ...
        'VLWRQ',[],'nuLWRQ',[],'deltaPart',[],'deltaLWRQIdentity',[], ...
        'croppedMass',[],'mode',[],'reconstructionTimeSeconds',[], ...
        'numCenters',[],'blocksPerDimension',[],'phaseRadiusBound',[],'study','');
end
function s=empty_summary_struct()
    s=struct('study','','h',[],'n',[],'halfWidth',[],'maxEta3D',[], ...
        'maxNuLWRQ',[],'maxDeltaLambda',[],'maxEtaUsing6DLambda',[], ...
        'maxPartitionError',[],'maxLWRQIdentityError',[],'numCenters',[], ...
        'blocksPerDimension',[],'phaseRadiusBound',[],'reconstructionTimeSeconds',[]);
end
function s=summarize_case(metrics,h,n,part,tRecon,study)
    eta=cellfun(@(x)x.eta3D,metrics); nu=cellfun(@(x)x.nuLWRQ,metrics);
    dl=cellfun(@(x)x.deltaLambda,metrics); e6=cellfun(@(x)x.etaUsing6DLambda,metrics);
    dp=cellfun(@(x)x.deltaPart,metrics); di=cellfun(@(x)x.deltaLWRQIdentity,metrics);
    s=empty_summary_struct(); s.study=study; s.h=h; s.n=n; s.halfWidth=h*n;
    s.maxEta3D=max(eta); s.maxNuLWRQ=max(nu); s.maxDeltaLambda=max(dl);
    s.maxEtaUsing6DLambda=max(e6); s.maxPartitionError=max(dp); s.maxLWRQIdentityError=max(di);
    s.reconstructionTimeSeconds=tRecon;
    if isempty(part)
        s.numCenters=NaN; s.blocksPerDimension=NaN; s.phaseRadiusBound=NaN;
    else
        s.numCenters=part.numCenters; s.blocksPerDimension=part.blocksPerDimension;
        s.phaseRadiusBound=part.rhoBound;
    end
end
function s=rmfield_if_present(s,names)
    for i=1:numel(names), if isfield(s,names{i}), s=rmfield(s,names{i}); end, end
end
