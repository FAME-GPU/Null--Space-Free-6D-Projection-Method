function results = run_fig1a_BC_method(method)
%RUN_FIG1A_BC_METHOD Run Method B or C for final Exp. 1 Fig. 1(a).
%
% N=5, 12 independent Bloch points.  Each completed point is checkpointed.
% Method B = nested reduced inverse Lanczos.
% Method C = explicit reduced inverse Lanczos.

    method = upper(char(method));
    if numel(method)~=1 || ~any(method=='BC')
        error('method must be ''B'' or ''C''.');
    end

    cfg = make_config_BC_fig1a();
    root = locate_root();
    outDir = fullfile(root,cfg.output.directory);
    ckDir = fullfile(outDir,cfg.output.checkpointDirectory);
    if ~exist(outDir,'dir'), mkdir(outDir); end
    if ~exist(ckDir,'dir'), mkdir(ckDir); end

    gpuInfo = gpu_initialize(cfg.gpu);
    path = build_exp1_12point_path(cfg);

    fprintf('\n============================================================\n');
    fprintf('Experiment 1 Fig. 1(a) -- Method %s\n',method);
    fprintf('N = %d, 12 independent Bloch points\n',cfg.problem.N);
    if method=='B'
        fprintf('Solver = inverse Lanczos + nested K_r/M CG\n');
    else
        fprintf('Solver = inverse Lanczos + explicit reduced inverse\n');
    end
    fprintf('============================================================\n\n');

    common = fig1a_build_experiment1_common(cfg,gpuInfo,cfg.problem.N);
    signature = make_BC_fig1a_signature(cfg,method,gpuInfo);
    qResults = cell(12,1);

    for iq = 1:12
        ck = fullfile(ckDir,sprintf('%s_q%02d.mat',method,iq));
        if exist(ck,'file')
            try
                S = load(ck,'qResult','signatureSaved');
                if isfield(S,'qResult') && isfield(S,'signatureSaved') && ...
                        isequaln(S.signatureSaved,signature)
                    qResults{iq} = S.qResult;
                    fprintf('[%s q%02d/12] checkpoint reused: outer=%d\n', ...
                        method,iq,S.qResult.outerSteps);
                    continue;
                end
                warning('Ignoring incompatible checkpoint: %s',ck);
            catch ME
                warning('Ignoring unreadable checkpoint %s (%s).',ck,ME.message);
            end
        end

        q = path.q(:,iq);
        fprintf('\n[%s q%02d/12] q=[%.8f %.8f %.8f]^T\n', ...
            method,iq,q(1),q(2),q(3));
        budget = make_run_budget(Inf,Inf);
        seed = cfg.lanczos.rngSeed + iq;

        if method=='B'
            sys = fig1a_build_experiment1_q_system_B_only(common,q,cfg);
            one = run_method_B_fig1(sys,cfg,seed,budget);
        else
            sys = build_experiment1_q_system(common,q,cfg);
            one = run_method_C_fig1(sys,cfg,seed,budget);
        end

        one.q = q;
        one.pointIndex = iq;
        one.pathCoordinate = path.pathCoordinate(iq);
        qResult = one; %#ok<NASGU>
        signatureSaved = signature; %#ok<NASGU>
        save(ck,'qResult','signatureSaved','-v7.3');
        qResults{iq} = one;

        fprintf('[%s q%02d DONE] outer=%d, linear=%d, wall=%.3f s\n', ...
            method,iq,one.outerSteps,one.totalLinearSolverIterations,one.totalWallSeconds);
        clear sys one qResult signatureSaved budget
    end

    results = struct();
    results.figure = '1a';
    results.method = method;
    results.N = cfg.problem.N;
    results.NF = common.NF;
    results.reducedDimension = 2*common.NF;
    results.massInfo = common.mass.info;
    results.path = path;
    results.qResults = qResults;
    results.cfg = cfg;
    results.gpuInfo = gpuInfo;
    results.matlabVersion = version;
    results.generatedAt = datestr(now,30);

    matFile = fullfile(outDir,sprintf('fig1a_%s.mat',method));
    csvFile = fullfile(outDir,sprintf('fig1a_%s.csv',method));
    txtFile = fullfile(outDir,sprintf('fig1a_%s_summary.txt',method));
    save(matFile,'results','-v7.3');
    write_csv(results,csvFile);
    write_summary(results,txtFile);

    fprintf('\nSaved Method %s Fig. 1(a) results:\n',method);
    fprintf('  %s\n  %s\n  %s\n',matFile,csvFile,txtFile);
end

function root = locate_root()
    root = fileparts(which('START_FIG1A_METHOD_B'));
    if isempty(root), root = fileparts(which('START_FIG1A_METHOD_C')); end
    if isempty(root), root = pwd; end
end

function write_csv(results,fileName)
    n = numel(results.qResults);
    point = (1:n)'; pathCoordinate = zeros(n,1);
    outerApplications = zeros(n,1); totalLinearIterations = zeros(n,1);
    wallSeconds = zeros(n,1);
    for i=1:n
        r=results.qResults{i};
        pathCoordinate(i)=r.pathCoordinate;
        outerApplications(i)=r.outerSteps;
        totalLinearIterations(i)=r.totalLinearSolverIterations;
        wallSeconds(i)=r.totalWallSeconds;
    end
    writetable(table(point,pathCoordinate,outerApplications, ...
        totalLinearIterations,wallSeconds),fileName);
end

function write_summary(results,fileName)
    fid=fopen(fileName,'w');
    if fid<0, error('Cannot open %s.',fileName); end
    c=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'Experiment 1 Fig. 1(a) -- Method %s\n',results.method);
    fprintf(fid,'N=%d, NF=%d\n\n',results.N,results.NF);
    fprintf(fid,'point  pathCoord  outerApps  linearIters  wallSeconds\n');
    for i=1:numel(results.qResults)
        r=results.qResults{i};
        fprintf(fid,'%2d  %.6f  %d  %d  %.6f\n', ...
            i,r.pathCoordinate,r.outerSteps,r.totalLinearSolverIterations,r.totalWallSeconds);
    end
end
