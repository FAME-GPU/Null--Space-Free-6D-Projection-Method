function results = run_fig1bc_method(method)
%RUN_FIG1BC_METHOD Generate Fig. 1(b,c) data for one independent method.
%
% N=3:8 at one fixed Bloch point. After the first non-success at N=N*, all
% larger N are skipped for that method. Limits are 1e6 total linear-solver
% iterations and 1e4 solver seconds. GPU OOM is an additional natural stop.

    method=upper(char(method));
    if numel(method)~=1 || ~any(method=='ABC'), error('method must be A, B, or C.'); end
    cfg=make_config_exp1bc(); Ns=cfg.scale.NList(:).';
    root=exp1bc_project_root();
    outDir=fullfile(root,cfg.output.resultsDirectory);
    ckDir=fullfile(root,cfg.output.checkpointDirectory);
    if ~exist(outDir,'dir'), mkdir(outDir); end
    if ~exist(ckDir,'dir'), mkdir(ckDir); end

    scaleResults=cell(numel(Ns),1); stopLarger=false;
    for ii=1:numel(Ns)
        N=Ns(ii); cfg.problem.N=N;
        ck=fullfile(ckDir,sprintf('fig1bc_%s_N%d.mat',method,N));

        % Resolve GPU identity for the checkpoint signature without resetting.
        g0=gpuDevice(cfg.gpu.deviceIndex);
        signature=make_exp1bc_signature(cfg,method,N);
        signature.gpuName=g0.Name; signature.matlabVersion=version;

        if stopLarger
            nResult=make_skipped(N,method,'skipped_after_previous_limit');
            save(ck,'nResult','signature'); scaleResults{ii}=nResult; continue
        end

        if exist(ck,'file')
            try
                S=load(ck,'nResult','signature');
                compatible=isfield(S,'signature') && isfield(S,'nResult') && ...
                    isfield(S.nResult,'status') && exp1_signature_equal(S.signature,signature);
            catch
                compatible=false;
            end
            if compatible && ~strcmp(S.nResult.status,'skipped_after_previous_limit')
                scaleResults{ii}=S.nResult;
                fprintf('[Fig1bc %s] reused N=%d status=%s\n',method,N,S.nResult.status);
                if ~strcmp(S.nResult.status,'success'), stopLarger=true; end
                continue
            elseif exist(ck,'file')
                warning('Ignoring incompatible/stale checkpoint %s.',ck);
            end
        end

        fprintf('\n============================================================\n');
        fprintf('[Fig1bc %s] N=%d, NF=%.0f\n',method,N,(2*N)^6);
        fprintf('============================================================\n');

        common=[]; sys=[]; one=[]; budget=[]; commonSeconds=NaN; massInfo=[];
        gpuAvailClean=NaN; gpuAvailAfterCommon=NaN; gpuAvailAfterSetup=NaN;
        status='success'; errMsg='';

        try
            gpuInfo=reset_gpu_for_scale(cfg.gpu,N,method);
            gpuAvailClean=gpuInfo.availableMemoryAtStart;

            % Common material/rank-1 construction is excluded from Fig. 1(c).
            commonTimer=tic;
            common=build_experiment1_common(cfg,gpuInfo,N);
            wait(gpuDevice); commonSeconds=toc(commonTimer);
            g=gpuDevice; gpuAvailAfterCommon=g.AvailableMemory;
            massInfo=common.mass.info;
            fprintf('[Fig1bc %s] common build %.3f s; GPU free after common %.3f GiB\n', ...
                method,commonSeconds,gpuAvailAfterCommon/2^30);

            epsMean=common.epsMean; epsMax=common.epsMax;
            budget=make_run_budget(cfg.budget.maxTotalLinearIterations,cfg.budget.maxSolverWallSeconds);
            budget.checkCanContinue('q/method setup start');

            if method=='A'
                sys=build_experiment1_q_system_A_only(common,cfg.scale.qPhys,cfg);
                lb=sys.minQq/epsMax;
                if cfg.shift.sigma>=lb, error('Unsafe shift at N=%d: sigma >= lower positive-spectrum bound.',N); end
            elseif method=='B'
                sys=build_experiment1_q_system_B_only(common,cfg.scale.qPhys,cfg);
            else
                sys=build_experiment1_q_system_C_compact(common,cfg.scale.qPhys,cfg);
            end
            budget.checkCanContinue('q/method setup complete');

            % sys owns the mass kernel/reference it needs. common is no longer
            % needed; clearing it releases the six Xi arrays before the solve.
            clear common
            g=gpuDevice; gpuAvailAfterSetup=g.AvailableMemory;
            fprintf('[Fig1bc %s] GPU free after compact method setup %.3f GiB\n', ...
                method,gpuAvailAfterSetup/2^30);

            if method=='A'
                one=run_method_A_fig1bc(sys,cfg,cfg.eigs.rngSeed,epsMean,budget);
            elseif method=='B'
                one=run_method_B_fig1bc(sys,cfg,cfg.lanczos.rngSeed+1,budget);
            else
                one=run_method_C_fig1bc(sys,cfg,cfg.lanczos.rngSeed+1,budget);
            end
            budget.check('completed eigensolve');
            if ~one.converged, error('EXP1:NotConverged','Method did not converge.'); end

        catch ME
            [status,controlled]=classify_exp1_exception(ME);
            errMsg=ME.message;
            if ~controlled, status='runtime_error'; end
        end

        if isempty(budget)
            snap=struct('iterationLimit',cfg.budget.maxTotalLinearIterations, ...
                'timeLimitSeconds',cfg.budget.maxSolverWallSeconds,'iterationsUsed',NaN, ...
                'elapsedSeconds',NaN,'remainingIterations',NaN);
        else
            snap=budget.snapshot();
        end

        if strcmp(status,'success')
            if one.totalLinearSolverIterations~=snap.iterationsUsed
                error('EXP1:CounterMismatch', ...
                    'Linear iteration counter mismatch at N=%d: solver=%g, budget=%g.', ...
                    N,one.totalLinearSolverIterations,snap.iterationsUsed);
            end
            nResult=one; nResult.status='success'; nResult.errorMessage='';
            nResult.solverWallSeconds=snap.elapsedSeconds;
            nResult.methodSetupSeconds=max(snap.elapsedSeconds-one.totalWallSeconds,0);
            nResult.plotLinearIterations=one.totalLinearSolverIterations;
            nResult.plotWallSeconds=snap.elapsedSeconds;
        else
            nResult=make_failure(N,method,status,errMsg,snap.iterationsUsed,snap.elapsedSeconds);
            if strcmp(status,'iteration_limit'), nResult.plotLinearIterations=cfg.budget.maxTotalLinearIterations; end
            if strcmp(status,'time_limit'), nResult.plotWallSeconds=cfg.budget.maxSolverWallSeconds; end
            stopLarger=true;
            warning('Fig1bc Method %s N=%d stopped with %s: %s',method,N,status,errMsg);
        end

        nResult.N=N; nResult.NF=(2*N)^6; nResult.method=method;
        nResult.commonBuildSeconds=commonSeconds; nResult.massInfo=massInfo; nResult.budget=snap;
        nResult.gpuAvailableCleanBytes=gpuAvailClean;
        nResult.gpuAvailableAfterCommonBytes=gpuAvailAfterCommon;
        nResult.gpuAvailableAfterMethodSetupBytes=gpuAvailAfterSetup;
        nResult.estimatedCPUOuterBasisGiB=estimate_outer_basis_gib(method,N,cfg);
        nResult.estimatedRank1KernelGiB=estimate_kernel_gib(N);

        save(ck,'nResult','signature','-v7.3'); scaleResults{ii}=nResult;
        fprintf('[Fig1bc %s] N=%d status=%s, linear=%.0f, wall=%.3f s\n', ...
            method,N,nResult.status,nResult.plotLinearIterations,nResult.plotWallSeconds);
        if method=='A' && isfield(nResult,'totalPlateauAccepts')
            fprintf('[Fig1bc A] plateau accepts=%d, max plateau residual=%.3e\n', ...
                nResult.totalPlateauAccepts,nResult.maxPlateauAcceptedResidual);
        end
        clear sys one budget common
    end

    results=struct('figure','1bc','method',method,'NList',Ns,'q',cfg.scale.qPhys, ...
        'scaleResults',{scaleResults},'cfg',cfg,'generatedAt',datestr(now,30));
    save(fullfile(outDir,sprintf('fig1bc_%s.mat',method)),'results','-v7.3');
    write_fig1bc_csv(results,fullfile(outDir,sprintf('fig1bc_%s.csv',method)));
    write_fig1bc_summary(results,fullfile(outDir,sprintf('fig1bc_%s.txt',method)));
end

function r=make_skipped(N,method,status)
    r=struct('N',N,'NF',(2*N)^6,'method',method,'status',status,'errorMessage','', ...
        'plotLinearIterations',NaN,'plotWallSeconds',NaN, ...
        'totalLinearSolverIterations',NaN,'totalWallSeconds',NaN);
end

function r=make_failure(N,method,status,msg,iters,wall)
    r=make_skipped(N,method,status); r.errorMessage=char(msg);
    r.observedLinearIterations=iters; r.observedWallSeconds=wall;
    r.plotLinearIterations=iters; r.plotWallSeconds=wall;
end

function g=estimate_outer_basis_gib(method,N,cfg)
    if method=='B' || method=='C'
        nr=2*(2*N)^6; g=nr*cfg.lanczos.krylovDimension*16/2^30;
    else
        g=NaN;
    end
end

function g=estimate_kernel_gib(N)
    B=4*N-1; M=(2*N)*B^5; g=M*16/2^30;
end

function write_fig1bc_csv(results,file)
    R=results.scaleResults; n=numel(R);
    N=zeros(n,1); NF=zeros(n,1); status=strings(n,1); lin=NaN(n,1); wall=NaN(n,1);
    observedLin=NaN(n,1); observedWall=NaN(n,1); outer=NaN(n,1);
    setup=NaN(n,1); plateau=NaN(n,1); maxPlateau=NaN(n,1); gpuFree=NaN(n,1);
    for i=1:n
        r=R{i}; N(i)=r.N; NF(i)=r.NF; status(i)=string(r.status);
        lin(i)=r.plotLinearIterations; wall(i)=r.plotWallSeconds;
        if isfield(r,'observedLinearIterations'), observedLin(i)=r.observedLinearIterations;
        elseif isfield(r,'totalLinearSolverIterations'), observedLin(i)=r.totalLinearSolverIterations; end
        if isfield(r,'observedWallSeconds'), observedWall(i)=r.observedWallSeconds;
        elseif isfield(r,'solverWallSeconds'), observedWall(i)=r.solverWallSeconds; end
        if isfield(r,'outerSteps'), outer(i)=r.outerSteps; end
        if isfield(r,'methodSetupSeconds'), setup(i)=r.methodSetupSeconds; end
        if isfield(r,'totalPlateauAccepts'), plateau(i)=r.totalPlateauAccepts; end
        if isfield(r,'maxPlateauAcceptedResidual'), maxPlateau(i)=r.maxPlateauAcceptedResidual; end
        if isfield(r,'gpuAvailableAfterMethodSetupBytes'), gpuFree(i)=r.gpuAvailableAfterMethodSetupBytes/2^30; end
    end
    writetable(table(N,NF,status,lin,wall,observedLin,observedWall,outer,setup,plateau,maxPlateau,gpuFree),file);
end

function write_fig1bc_summary(results,file)
    fid=fopen(file,'w'); if fid<0,error('Cannot open %s.',file);end
    c=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'Fig. 1(b,c) Method %s, q=[%.8f %.8f %.8f]^T\n',results.method,results.q);
    fprintf(fid,'Iteration limit = %.0f, time limit = %.0f s\n', ...
        results.cfg.budget.maxTotalLinearIterations,results.cfg.budget.maxSolverWallSeconds);
    fprintf(fid,'N  NF  status  plotLinearIterations  plotWallSeconds\n');
    for i=1:numel(results.scaleResults)
        r=results.scaleResults{i};
        fprintf(fid,'%d  %.0f  %s  %.12g  %.12g\n',r.N,r.NF,r.status,r.plotLinearIterations,r.plotWallSeconds);
        if isfield(r,'totalPlateauAccepts')
            fprintf(fid,'  A plateau accepts=%d, max residual=%.3e\n',r.totalPlateauAccepts,r.maxPlateauAcceptedResidual);
        end
        if isfield(r,'gpuAvailableAfterMethodSetupBytes') && isfinite(r.gpuAvailableAfterMethodSetupBytes)
            fprintf(fid,'  GPU free after method setup=%.3f GiB\n',r.gpuAvailableAfterMethodSetupBytes/2^30);
        end
        if isfield(r,'errorMessage') && ~isempty(r.errorMessage), fprintf(fid,'  reason: %s\n',r.errorMessage); end
    end
end
