function results = run_exp4_medium(mediumName)
%RUN_EXP4_MEDIUM Complete final production run for one Exp4 medium.
%
% Run Medium A and Medium B as separate scheduler jobs.  Each q point is
% checkpointed independently.  Only the four representative q points save
% one selected Fourier eigenvector for subsequent 3D LWRQ/field recovery.

    root=setup_exp4();
    cfg=make_config_exp4(mediumName);
    path=build_exp4_path(cfg);
    gpuInfo=gpu_initialize(cfg.gpu);

    fprintf('\n============================================================\n');
    fprintf('Experiment 4 FINAL GPU: %s\n',cfg.material.label);
    fprintf('N=%d, NF=%d, %d independent q points, %d plotted points, nev=%d\n', ...
        cfg.problem.N,(2*cfg.problem.N)^6,path.numIndependent,path.numPlot,cfg.spectrum.nev);
    fprintf('Selected pairs: ');
    for j=1:numel(cfg.selected.labels)
        fprintf('%s=(%d,%d)',cfg.selected.labels{j}, ...
            cfg.selected.pointIndices(j),cfg.selected.modeIndices(j));
        if j<numel(cfg.selected.labels), fprintf(', '); else, fprintf('\n'); end
    end
    fprintf('============================================================\n\n');

    totalTimer=tic;
    st=build_exp4_medium_static(cfg,gpuInfo);

    nQ=path.numIndependent;
    nev=cfg.spectrum.nev;
    lambdaIndependent=nan(nev,nQ);
    qTiming=nan(nQ,1); outerSteps=nan(nQ,1); cgAvg=nan(nQ,1);
    maxRitz=nan(nQ,1); loadedFromCheckpoint=false(nQ,1);

    ckDir=fullfile(root,'checkpoint');
    completedThisRun=0; elapsedSolve=0;

    for iq=1:nQ
        q=path.qIndependent(:,iq);
        jsel=find(cfg.selected.pointIndices==iq,1);
        if isempty(jsel), selectedMode=0; else, selectedMode=cfg.selected.modeIndices(jsel); end
        sig=exp4_checkpoint_signature(cfg,q,iq,selectedMode);
        ckFile=fullfile(ckDir,sprintf('%s_q%03d.mat',cfg.output.tag,iq));

        useCk=false;
        if cfg.run.reuseQCheckpoints && exist(ckFile,'file')
            C=load(ckFile,'qres','signature');
            modeOK=isfield(C,'qres') && isfield(C.qres,'selectedMode') && ...
                double(C.qres.selectedMode)==double(selectedMode);
            if selectedMode>0
                modeOK=modeOK && isfield(C.qres,'Xselected') && ~isempty(C.qres.Xselected) && ...
                    isfield(C.qres,'selectedLambda') && ~isempty(C.qres.selectedLambda);
            end
            residualOK=isfield(C,'qres') && isfield(C.qres,'converged') && C.qres.converged && ...
                isfield(C.qres,'maxRelativeInverseRitzResidual') && ...
                isfinite(C.qres.maxRelativeInverseRitzResidual) && ...
                C.qres.maxRelativeInverseRitzResidual<=cfg.solver.tolLanczos;
            if isfield(C,'signature') && signatures_equal_exp4(C.signature,sig) && modeOK && residualOK
                qres=C.qres; useCk=true;
            else
                warning('Ignoring incompatible checkpoint: %s',ckFile);
            end
        end

        if useCk
            loadedFromCheckpoint(iq)=true;
            fprintf('[%s q %02d/%02d] checkpoint, lambda1=%.6e, lambda%d=%.6e', ...
                cfg.output.tag,iq,nQ,qres.lambda(1),nev,qres.lambda(end));
            if selectedMode>0, fprintf(', selected mode=%d',selectedMode); end
            fprintf('\n');
        else
            fprintf('[%s q %02d/%02d] solving q=[%.5f %.5f %.5f]^T', ...
                cfg.output.tag,iq,nQ,q(1),q(2),q(3));
            if selectedMode>0, fprintf(' with selected mode %d',selectedMode); end
            fprintf(' ...\n');

            qres=solve_exp4_qpoint(st,cfg,q,iq,selectedMode);
            fprintf('  outer=%d, wall=%.1f s, Mhat-CG avg=%.2f, spectrum=[%.6e, %.6e]\n', ...
                qres.outerSteps,qres.wallTimeSeconds,qres.cgAverageIterations, ...
                qres.lambda(1),qres.lambda(end));

            if cfg.run.saveQCheckpoints
                signature=sig; %#ok<NASGU>
                save(ckFile,'qres','signature','-v7.3');
            end
            completedThisRun=completedThisRun+1;
            elapsedSolve=elapsedSolve+qres.wallTimeSeconds;
            if completedThisRun>=2
                avg=elapsedSolve/completedThisRun;
                fprintf('  current fresh-point average %.1f s; nominal %d-point projection %.2f h\n', ...
                    avg,nQ,nQ*avg/3600);
            end
        end

        if numel(qres.lambda)~=nev, error('Checkpoint/result eigenvalue count mismatch.'); end
        if qres.selectedMode~=selectedMode
            error('Selected-mode mismatch at point %d: got %d, expected %d.', ...
                iq,qres.selectedMode,selectedMode);
        end
        if selectedMode>0 && isempty(qres.Xselected)
            error('Selected q checkpoint lacks mode %d eigenvector at point %d.',selectedMode,iq);
        end

        lambdaIndependent(:,iq)=qres.lambda(:);
        qTiming(iq)=qres.wallTimeSeconds;
        outerSteps(iq)=qres.outerSteps;
        cgAvg(iq)=qres.cgAverageIterations;
        maxRitz(iq)=qres.maxRelativeInverseRitzResidual;
        clear qres C
    end

    lambdaPlot=[lambdaIndependent lambdaIndependent(:,1)];

    %% Four 3D LWRQ / visualization postprocesses.
    ns=numel(cfg.selected.labels);
    selected=cell(1,ns);
    for j=1:ns
        iq=cfg.selected.pointIndices(j);
        desiredMode=cfg.selected.modeIndices(j);
        q=path.qIndependent(:,iq);
        qCk=fullfile(ckDir,sprintf('%s_q%03d.mat',cfg.output.tag,iq));
        C=load(qCk,'qres','signature');
        expected=exp4_checkpoint_signature(cfg,q,iq,desiredMode);
        if ~signatures_equal_exp4(C.signature,expected) || ...
                ~isfield(C.qres,'selectedMode') || C.qres.selectedMode~=desiredMode || ...
                isempty(C.qres.Xselected)
            error('Selected q checkpoint became incompatible: %s',qCk);
        end

        postSig=exp4_post_signature(cfg,j,C.qres.selectedLambda);
        postFile=fullfile(ckDir,sprintf('%s_%s_post.mat',cfg.output.tag,cfg.selected.labels{j}));
        usePost=false;
        if cfg.run.reusePostCheckpoints && exist(postFile,'file')
            P=load(postFile,'post','postSignature');
            if isfield(P,'postSignature') && isequaln(P.postSignature,postSig) && ...
                    isfield(P,'post') && P.post.modeIndex==desiredMode
                post=P.post; usePost=true;
            end
        end

        if usePost
            fprintf('[%s %s] reusing LWRQ/field-slice checkpoint.\n',cfg.output.tag,cfg.selected.labels{j});
        else
            fprintf('[%s %s] 3D LWRQ + z0/x0/xy field slices for mode %d ...\n', ...
                cfg.output.tag,cfg.selected.labels{j},desiredMode);
            post=postprocess_selected_exp4(st,cfg,C.qres,cfg.selected.labels{j});
            if cfg.run.savePostCheckpoints
                postSignature=postSig; %#ok<NASGU>
                save(postFile,'post','postSignature','-v7.3');
            end
        end
        selected{j}=post;
        fprintf('  point=%d mode=%d: lambda6D=%.12e, lambdaLWRQ=%.12e, rel.diff=%.3e, eta=%.3e, nu=%.3e\n', ...
            post.pointIndex,post.modeIndex,post.lambda6D,post.lambdaLWRQ, ...
            post.deltaLambda,post.eta3D,post.nuLWRQ);
        clear C P post
    end

    results=struct();
    results.version='Exp4_Public_v2_ElementwiseRitz_AxialMaterial';
    results.cfg=cfg;
    results.material=cfg.material;
    results.materialInfo=st.materialInfo;
    results.path=path;
    results.spectrum=struct('lambdaIndependent',lambdaIndependent,'lambdaPlot',lambdaPlot);
    results.solverStats=struct('wallTimeSeconds',qTiming,'outerSteps',outerSteps, ...
        'mhatAverageIterations',cgAvg,'maxRelativeInverseRitzResidual',maxRitz, ...
        'loadedFromCheckpoint',loadedFromCheckpoint);
    results.selected=selected;
    results.totalWallSeconds=toc(totalTimer);
    results.gpu=rmfield(gpuInfo,intersect(fieldnames(gpuInfo),{'availableMemoryAtStart'}));

    outFile=fullfile(root,'output',cfg.output.resultFile);
    save(outFile,'results','-v7.3');
    write_exp4_outputs(results,root);

    fprintf('\n============================================================\n');
    fprintf('Experiment 4 %s COMPLETE: current MATLAB wall %.2f h\n', ...
        cfg.output.tag,results.totalWallSeconds/3600);
    fprintf('Results: %s\n',outFile);
    fprintf('============================================================\n');
end
