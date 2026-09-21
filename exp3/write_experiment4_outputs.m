function write_experiment4_outputs(results,root)
%WRITE_EXPERIMENT4_OUTPUTS Save compact MAT, CSV summaries and manuscript values.

    cfg=results.cfg;
    out=fullfile(root,'output');
    if ~exist(out,'dir'), mkdir(out); end

    save(fullfile(out,cfg.output.resultFile),'results','-v7.3');

    allCases=[results.meshCases(:); results.windowCases(:)];
    writetable(case_table(allCases),fullfile(out,cfg.output.detailedCSV));
    writetable(summary_table(results.meshSummary),fullfile(out,cfg.output.meshCSV));
    writetable(summary_table(results.windowSummary),fullfile(out,cfg.output.windowCSV));

    isDefault=arrayfun(@(s)strcmp(s.study,'mesh') && ...
        abs(s.h-cfg.default.h)<1e-14 && s.n==cfg.default.n,results.meshCases);
    defaultCases=results.meshCases(isDefault);
    writetable(case_table(defaultCases),fullfile(out,cfg.output.defaultCSV));

    write_summary_txt(results,fullfile(out,cfg.output.summaryTXT),defaultCases);
    write_latex_values(results,fullfile(out,cfg.output.latexTXT),defaultCases);
end

function T=case_table(S)
    mode=[S.mode].'; h=[S.h].'; n=[S.n].'; halfWidth=[S.halfWidth].';
    lambda6D=[S.lambda6D].'; lambda3D=[S.lambda3D].'; lambdaCrop=[S.lambdaCrop].';
    eta3D=[S.eta3D].'; nuLWRQ=[S.nuLWRQ].'; VLWRQ=[S.VLWRQ].';
    deltaLambda=[S.deltaLambda].'; etaUsing6DLambda=[S.etaUsing6DLambda].';
    deltaPart=[S.deltaPart].'; deltaLWRQIdentity=[S.deltaLWRQIdentity].';
    numCenters=[S.numCenters].'; blocksPerDimension=[S.blocksPerDimension].';
    phaseRadiusBound=[S.phaseRadiusBound].';
    reconstructionTimeSeconds=[S.reconstructionTimeSeconds].';
    study=string({S.study}.');
    T=table(study,mode,h,n,halfWidth,lambda6D,lambda3D,lambdaCrop,eta3D, ...
        nuLWRQ,VLWRQ,deltaLambda,etaUsing6DLambda,deltaPart,deltaLWRQIdentity, ...
        numCenters,blocksPerDimension,phaseRadiusBound,reconstructionTimeSeconds);
end

function T=summary_table(S)
    study=string({S.study}.'); h=[S.h].'; n=[S.n].'; halfWidth=[S.halfWidth].';
    maxEta3D=[S.maxEta3D].'; maxNuLWRQ=[S.maxNuLWRQ].';
    maxDeltaLambda=[S.maxDeltaLambda].'; maxEtaUsing6DLambda=[S.maxEtaUsing6DLambda].';
    maxPartitionError=[S.maxPartitionError].'; maxLWRQIdentityError=[S.maxLWRQIdentityError].';
    numCenters=[S.numCenters].'; blocksPerDimension=[S.blocksPerDimension].';
    phaseRadiusBound=[S.phaseRadiusBound].'; reconstructionTimeSeconds=[S.reconstructionTimeSeconds].';
    T=table(study,h,n,halfWidth,maxEta3D,maxNuLWRQ,maxDeltaLambda, ...
        maxEtaUsing6DLambda,maxPartitionError,maxLWRQIdentityError,numCenters, ...
        blocksPerDimension,phaseRadiusBound,reconstructionTimeSeconds);
end

function write_summary_txt(results,path,defaultCases)
    cfg=results.cfg; e=results.eigen;
    fid=fopen(path,'w'); if fid<0, error('Cannot open summary file.'); end
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>

    fprintf(fid,'Experiment 3: N=5 tracked-mode physical-space closure and LWRQ validation (all GPU)\n');
    fprintf(fid,'============================================================================\n');
    fprintf(fid,'q = [%.8f %.8f %.8f]^T\n',cfg.problem.q);
    fprintf(fid,'N = %d, NF = %d, reduced dimension = %d\n',cfg.problem.N,e.NF,2*e.NF);
    fprintf(fid,'Tracked N=5 modes = [%d, %d]\n',e.modes(1),e.modes(2));
    fprintf(fid,'Tracking origin: N=2 modes [%d,%d]\n',cfg.tracking.n2Modes);
    fprintf(fid,'Fourier overlaps = [%.8f, %.8f]\n',cfg.tracking.fourierOverlap);
    fprintf(fid,'Pair principal cosines = [%.8f, %.8f]\n\n',cfg.tracking.pairPrincipalCosines);

    fprintf(fid,'Selected N=5 eigenvalues = [%.15e, %.15e]\n',e.lambda6D);
    fprintf(fid,'Selected-eigenpair source = %s\n',results.eigenpairSource);
    fprintf(fid,'Lanczos scan k=%d, outer steps=%d, wall=%.3f s, Mhat-CG avg=%.3f\n', ...
        cfg.scan.N5.k,e.outerSteps,e.scanWallTimeSeconds,e.MhatCGAverageIterations);
    fprintf(fid,'Selected inverse-Ritz relative residuals = [%.3e, %.3e]\n', ...
        e.selectedRelativeInverseRitzResidual);
    fprintf(fid,'Recovery M-CG relative residuals = [%.3e, %.3e]\n\n',e.recoveryRelativeResiduals);

    fprintf(fid,'Taylor p=%d, phase-radius target=%.3f, Qalpha build=%.3f s\n', ...
        cfg.taylor.order,cfg.taylor.phaseRadiusTarget,results.taylorBasisBuildTimeSeconds);
    fprintf(fid,'Default Yee h=%.5f, n=%d, half-width=%.3f\n', ...
        cfg.default.h,cfg.default.n,cfg.default.h*cfg.default.n);
    W=results.windowMasterReconstruction;
    fprintf(fid,'Window master h=%.5f, n=%d, half-width=%.3f, centers=%d (%d^3), rho<=%.4f\n', ...
        W.h,W.n,W.halfWidth,W.numCenters,W.blocksPerDimension,W.phaseRadiusBound);
    fprintf(fid,'Window master per-mode reconstruction times = %s s\n\n',mat2str(W.perModeTimeSeconds,6));

    fprintf(fid,'Default tracked-mode closure metrics\n');
    for i=1:numel(defaultCases)
        s=defaultCases(i);
        fprintf(fid,['  mode %d: lambda6D=%.15e, lambda3D=%.15e, eta3D=%.6e, ', ...
            'nuLWRQ=%.6e, V_LWRQ=%.6e, deltaLambda=%.6e\n'], ...
            s.mode,s.lambda6D,s.lambda3D,s.eta3D,s.nuLWRQ,s.VLWRQ,s.deltaLambda);
        fprintf(fid,'           eta(lambda6D)=%.6e, partition=%.3e, LWRQ identity=%.3e\n', ...
            s.etaUsing6DLambda,s.deltaPart,s.deltaLWRQIdentity);
    end

    fprintf(fid,'\nMesh refinement worst-case over modes 11 and 114\n');
    for i=1:numel(results.meshSummary)
        s=results.meshSummary(i);
        fprintf(fid,['  h=%g n=%d: max eta3D=%.6e, max nu=%.6e, max deltaLambda=%.6e, ', ...
            'centers=%d, rho<=%.4f, recon=%.2f s\n'], ...
            s.h,s.n,s.maxEta3D,s.maxNuLWRQ,s.maxDeltaLambda, ...
            s.numCenters,s.phaseRadiusBound,s.reconstructionTimeSeconds);
    end

    fprintf(fid,'\nWindow sensitivity (exact crops of the same h=0.025, L=4 master field)\n');
    for i=1:numel(results.windowSummary)
        s=results.windowSummary(i);
        fprintf(fid,'  L=%.3f n=%d: max eta3D=%.6e, max nu=%.6e, max deltaLambda=%.6e\n', ...
            s.halfWidth,s.n,s.maxEta3D,s.maxNuLWRQ,s.maxDeltaLambda);
    end
    fprintf(fid,'\nFig. 3(b) spatial slices are saved at h=0.025, n=160, L=4 on z=0 and x=0.\n');
    fprintf(fid,'Total experiment wall time = %.3f s = %.3f min\n', ...
        results.totalWallTimeSeconds,results.totalWallTimeSeconds/60);
end

function write_latex_values(results,path,defaultCases)
    fid=fopen(path,'w'); if fid<0, error('Cannot open LaTeX-value file.'); end
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%% Experiment 3 all-GPU measured values for manuscript insertion\n');
    for i=1:numel(defaultCases)
        s=defaultCases(i);
        fprintf(fid,'mode_%d_lambda6D = %.8e\n',s.mode,s.lambda6D);
        fprintf(fid,'mode_%d_lambda3D = %.8e\n',s.mode,s.lambda3D);
        fprintf(fid,'mode_%d_eta3D = %.8e\n',s.mode,s.eta3D);
        fprintf(fid,'mode_%d_nuLWRQ = %.8e\n',s.mode,s.nuLWRQ);
        fprintf(fid,'mode_%d_deltaLambda = %.8e\n',s.mode,s.deltaLambda);
    end
    D=results.meshSummary;
    [~,iFine]=min([D.h]);
    fprintf(fid,'default_max_eta3D = %.8e\n',max([defaultCases.eta3D]));
    fprintf(fid,'default_max_nuLWRQ = %.8e\n',max([defaultCases.nuLWRQ]));
    fprintf(fid,'default_max_deltaLambda = %.8e\n',max([defaultCases.deltaLambda]));
    fprintf(fid,'finest_max_eta3D = %.8e\n',D(iFine).maxEta3D);
    fprintf(fid,'finest_max_nuLWRQ = %.8e\n',D(iFine).maxNuLWRQ);
    fprintf(fid,'finest_max_deltaLambda = %.8e\n',D(iFine).maxDeltaLambda);
    fprintf(fid,'max_partition_error = %.8e\n',max([D.maxPartitionError]));
    fprintf(fid,'max_lwrq_identity_error = %.8e\n',max([D.maxLWRQIdentityError]));

    W=results.windowSummary;
    [~,iMax]=max([W.halfWidth]);
    fprintf(fid,'window_Lmax = %.8e\n',W(iMax).halfWidth);
    fprintf(fid,'window_Lmax_max_eta3D = %.8e\n',W(iMax).maxEta3D);
    fprintf(fid,'window_Lmax_max_nuLWRQ = %.8e\n',W(iMax).maxNuLWRQ);
    fprintf(fid,'window_Lmax_max_deltaLambda = %.8e\n',W(iMax).maxDeltaLambda);
end
