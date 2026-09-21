function results = run_fig1a_A_only()
%RUN_FIG1A_A_ONLY N=5 Method-A sweep over the 12 independent Bloch points.
%
% Each completed Bloch point is checkpointed independently.  Restarting the
% driver reuses only checkpoints with an exactly matching solver/platform
% signature.

    cfg = make_config_A_fig1a();
    root = fileparts(which('START_FIG1A_METHOD_A'));
    if isempty(root), root = pwd; end
    outDir = fullfile(root,cfg.output.directory);
    ckDir = fullfile(outDir,cfg.output.checkpointDirectory);
    if ~exist(outDir,'dir'), mkdir(outDir); end
    if ~exist(ckDir,'dir'), mkdir(ckDir); end

    gpuInfo = gpu_initialize(cfg.gpu);
    path = build_exp1_12point_path(cfg);

    fprintf('\n============================================================\n');
    fprintf('Experiment 1 Fig. 1(a) -- Method A only\n');
    fprintf('N = %d, 12 independent Bloch points\n',cfg.problem.N);
    fprintf('Outer: MATLAB eigs on CPU\n');
    fprintf('Inner: GPU preconditioned MINRES + guarded correction\n');
    fprintf('Target MINRES tolerance      = %.1e\n',cfg.shift.minresTolerance);
    fprintf('Plateau acceptance ceiling   = %.1e\n',cfg.shift.plateauAcceptanceCeiling);
    fprintf('Plateau failed cycles needed = %d\n',cfg.shift.plateauMinFailedRecoveryCycles);
    fprintf('============================================================\n\n');

    wait(gpuDevice);
    commonTimer = tic;
    common = fig1a_build_experiment1_common(cfg,gpuInfo,cfg.problem.N);
    wait(gpuDevice);
    commonBuildSeconds = toc(commonTimer);
    epsMean = common.epsMean;

    fprintf('[common] NF=%d, original dimension=%d, build=%.3f s\n', ...
        common.NF,common.ndof,commonBuildSeconds);
    fprintf('[material] eps min/max/mean = %.6g / %.6g / %.6g\n', ...
        common.epsMin,common.epsMax,common.epsMean);
    fprintf('[rank-1] M=%g, B=%g, exactCertified=%d\n\n', ...
        common.mass.info.M,common.mass.info.B,common.mass.info.exactCertified);

    signature = make_A_fig1a_signature(cfg,gpuInfo);
    qResults = cell(12,1);

    for iq = 1:12
        ck = fullfile(ckDir,sprintf('A_q%02d.mat',iq));
        if exist(ck,'file')
            try
                S = load(ck,'qResult','signatureSaved');
                if isfield(S,'qResult') && isfield(S,'signatureSaved') && ...
                        A_fig1a_signature_equal(S.signatureSaved,signature)
                    qResults{iq} = S.qResult;
                    fprintf('[A q%02d/12] checkpoint reused: outer=%d, MINRES=%d, plateau=%d\n', ...
                        iq,S.qResult.outerSteps,S.qResult.totalLinearSolverIterations, ...
                        S.qResult.totalPlateauAccepts);
                    continue;
                end
                warning('Ignoring incompatible checkpoint: %s',ck);
            catch ME
                warning('Ignoring unreadable checkpoint %s (%s).',ck,ME.message);
            end
        end

        q = path.q(:,iq);
        fprintf('\n[A q%02d/12] q=[%.8f %.8f %.8f]^T\n',iq,q(1),q(2),q(3));

        sys = build_experiment1_q_system_A_only(common,q,cfg);
        positiveLowerBound = sys.minQq/common.epsMax;
        if cfg.shift.sigma >= positiveLowerBound
            error('Unsafe shift at q%02d: sigma=%.3e >= lower bound %.3e.', ...
                iq,cfg.shift.sigma,positiveLowerBound);
        end

        one = run_method_A_fig1a(sys,cfg,cfg.eigs.rngSeed+iq-1,epsMean);
        one.q = q;
        one.pointIndex = iq;
        one.pathCoordinate = path.pathCoordinate(iq);
        one.positiveSpectrumLowerBound = positiveLowerBound;

        fprintf(['[A q%02d DONE] converged=%d, outer=%d, total MINRES=%d, ', ...
                 'plateau accepts=%d, max plateau=%.3e, max GEVP residual=%.3e, ', ...
                 'eigs=%.3f s\n'], ...
                iq,one.converged,one.outerSteps,one.totalLinearSolverIterations, ...
                one.totalPlateauAccepts,one.maxPlateauAcceptedResidual, ...
                one.maxOriginalGEVPResidual,one.eigensolveSeconds);
        if ~one.converged
            warning(['Bloch point q%02d did not satisfy the final eigs success test. ', ...
                     'The point is saved for diagnosis and the sweep continues.'],iq);
        end

        qResult = one; %#ok<NASGU>
        signatureSaved = signature; %#ok<NASGU>
        save(ck,'qResult','signatureSaved','-v7.3');
        qResults{iq} = one;
        clear sys one qResult signatureSaved
    end

    results = struct();
    results.figure = '1a';
    results.method = 'A';
    results.description = 'CPU eigs + GPU preconditioned MINRES with plateau fallback';
    results.N = cfg.problem.N;
    results.NF = common.NF;
    results.originalDimension = common.ndof;
    results.commonBuildSeconds = commonBuildSeconds;
    results.massInfo = common.mass.info;
    results.path = path;
    results.qResults = qResults;
    results.cfg = cfg;
    results.gpuInfo = gpuInfo;
    results.matlabVersion = version;
    results.generatedAt = datestr(now,30);

    save(fullfile(outDir,cfg.output.matFile),'results','-v7.3');
    write_A_fig1a_csv(results,fullfile(outDir,cfg.output.csvFile));
    write_A_fig1a_summary(results,fullfile(outDir,cfg.output.summaryFile));

    fprintf('\n============================================================\n');
    fprintf('A-only Fig. 1(a) sweep completed.\n');
    fprintf('Saved: %s\n',fullfile(outDir,cfg.output.matFile));
    fprintf('       %s\n',fullfile(outDir,cfg.output.csvFile));
    fprintf('       %s\n',fullfile(outDir,cfg.output.summaryFile));
    fprintf('============================================================\n');
end

function write_A_fig1a_csv(results,fileName)
    qR = results.qResults;
    n = numel(qR);
    point = (1:n)';
    pathCoordinate = zeros(n,1);
    q1 = zeros(n,1); q2 = zeros(n,1); q3 = zeros(n,1);
    outerApps = zeros(n,1);
    totalMINRESIterations = zeros(n,1);
    plateauAccepts = zeros(n,1);
    maxPlateauResidual = NaN(n,1);
    maxInnerResidual = NaN(n,1);
    maxGEVPResidual = NaN(n,1);
    eigsSeconds = zeros(n,1);
    converged = false(n,1);

    for i = 1:n
        r = qR{i};
        pathCoordinate(i) = r.pathCoordinate;
        q1(i) = r.q(1); q2(i) = r.q(2); q3(i) = r.q(3);
        outerApps(i) = r.outerSteps;
        totalMINRESIterations(i) = r.totalLinearSolverIterations;
        plateauAccepts(i) = r.totalPlateauAccepts;
        maxPlateauResidual(i) = r.maxPlateauAcceptedResidual;
        maxInnerResidual(i) = r.maxInnerFinalRelativeResidual;
        maxGEVPResidual(i) = r.maxOriginalGEVPResidual;
        eigsSeconds(i) = r.eigensolveSeconds;
        converged(i) = r.converged;
    end

    T = table(point,pathCoordinate,q1,q2,q3,outerApps, ...
        totalMINRESIterations,plateauAccepts,maxPlateauResidual, ...
        maxInnerResidual,maxGEVPResidual,eigsSeconds,converged);
    writetable(T,fileName);
end

function write_A_fig1a_summary(results,fileName)
    fid = fopen(fileName,'w');
    if fid < 0, error('Cannot open summary file: %s',fileName); end
    c = onCleanup(@()fclose(fid)); %#ok<NASGU>

    cfg = results.cfg;
    fprintf(fid,'Experiment 1 Fig. 1(a) -- Method A only\n');
    fprintf(fid,'============================================================\n');
    fprintf(fid,'MATLAB = %s\n',results.matlabVersion);
    fprintf(fid,'GPU = %s\n',results.gpuInfo.name);
    fprintf(fid,'N = %d, NF = %d, original dimension = %d\n', ...
        results.N,results.NF,results.originalDimension);
    fprintf(fid,'Outer = MATLAB eigs on CPU\n');
    fprintf(fid,'Inner = GPU preconditioned MINRES\n');
    fprintf(fid,'MINRES target tolerance = %.3e\n',cfg.shift.minresTolerance);
    fprintf(fid,'Plateau acceptance ceiling = %.3e\n',cfg.shift.plateauAcceptanceCeiling);
    fprintf(fid,'Plateau failed recovery cycles required = %d\n', ...
        cfg.shift.plateauMinFailedRecoveryCycles);
    fprintf(fid,'Plateau material reduction threshold = %.3e\n\n', ...
        cfg.shift.plateauMinRelativeReduction);

    fprintf(fid,['point  pathCoord  outerApps  MINRESiters  plateauAccepts  ', ...
                 'maxPlateauRelres  maxInnerRelres  maxGEVPres  eigsSeconds  ok\n']);
    for i = 1:numel(results.qResults)
        r = results.qResults{i};
        fprintf(fid,'%2d  %.6f  %d  %d  %d  %.6e  %.6e  %.6e  %.6f  %d\n', ...
            i,r.pathCoordinate,r.outerSteps,r.totalLinearSolverIterations, ...
            r.totalPlateauAccepts,r.maxPlateauAcceptedResidual, ...
            r.maxInnerFinalRelativeResidual,r.maxOriginalGEVPResidual, ...
            r.eigensolveSeconds,r.converged);
    end

    totalPlateau = sum(cellfun(@(r)r.totalPlateauAccepts,results.qResults));
    plateauVals = cellfun(@(r)r.maxPlateauAcceptedResidual,results.qResults);
    maxPlateau = max(plateauVals,[],'omitnan');
    gevpVals = cellfun(@(r)r.maxOriginalGEVPResidual,results.qResults);
    maxGEVP = max(gevpVals,[],'omitnan');
    meanOuter = mean(cellfun(@(r)r.outerSteps,results.qResults));
    meanMINRES = mean(cellfun(@(r)r.totalLinearSolverIterations,results.qResults));

    fprintf(fid,'\nAggregate diagnostics\n');
    fprintf(fid,'---------------------\n');
    fprintf(fid,'mean outer applications = %.6f\n',meanOuter);
    fprintf(fid,'mean total MINRES iterations = %.6f\n',meanMINRES);
    fprintf(fid,'total plateau accepts = %d\n',totalPlateau);
    fprintf(fid,'maximum plateau-accepted residual = %.6e\n',maxPlateau);
    fprintf(fid,'maximum final original-GEVP residual = %.6e\n',maxGEVP);
end
