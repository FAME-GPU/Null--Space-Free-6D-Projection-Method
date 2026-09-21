function write_exp4_outputs(results,root)
%WRITE_EXP4_OUTPUTS Human-readable summaries and CSV exports.
    cfg=results.cfg; outDir=fullfile(root,'output');

    % Spectrum CSV: one row per plotted q and mode.
    nPlot=results.path.numPlot; nev=cfg.spectrum.nev;
    rows=nPlot*nev;
    point=zeros(rows,1); mode=zeros(rows,1); q1=zeros(rows,1); q2=q1; q3=q1; lambda=q1;
    c=0;
    for ip=1:nPlot
        for j=1:nev
            c=c+1; point(c)=ip; mode(c)=j;
            q1(c)=results.path.qPlot(1,ip); q2(c)=results.path.qPlot(2,ip); q3(c)=results.path.qPlot(3,ip);
            lambda(c)=results.spectrum.lambdaPlot(j,ip);
        end
    end
    writetable(table(point,mode,q1,q2,q3,lambda),fullfile(outDir,cfg.output.spectrumCSV));

    % Selected LWRQ CSV.
    ns=numel(results.selected);
    label=cell(ns,1); point=zeros(ns,1); mode=zeros(ns,1);
    lambda6D=zeros(ns,1); lambdaLWRQ=zeros(ns,1); relDiff=zeros(ns,1);
    eta3D=zeros(ns,1); nuLWRQ=zeros(ns,1); identity=zeros(ns,1);
    for j=1:ns
        p=results.selected{j}; label{j}=p.label; point(j)=p.pointIndex; mode(j)=p.modeIndex;
        lambda6D(j)=p.lambda6D; lambdaLWRQ(j)=p.lambdaLWRQ; relDiff(j)=p.deltaLambda;
        eta3D(j)=p.eta3D; nuLWRQ(j)=p.nuLWRQ; identity(j)=p.lwrqIdentityError;
    end
    writetable(table(label,point,mode,lambda6D,lambdaLWRQ,relDiff,eta3D,nuLWRQ,identity), ...
        fullfile(outDir,cfg.output.selectedCSV));

    % q timing CSV.
    qpoint=(1:results.path.numIndependent).';
    wall_s=results.solverStats.wallTimeSeconds(:);
    outer=results.solverStats.outerSteps(:);
    mhat_avg=results.solverStats.mhatAverageIterations(:);
    max_ritz=results.solverStats.maxRelativeInverseRitzResidual(:);
    from_checkpoint=results.solverStats.loadedFromCheckpoint(:);
    writetable(table(qpoint,wall_s,outer,mhat_avg,max_ritz,from_checkpoint), ...
        fullfile(outDir,cfg.output.timingCSV));

    % Text summary.
    fid=fopen(fullfile(outDir,cfg.output.summaryTXT),'w');
    if fid<0, error('Cannot open summary file.'); end
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'Experiment 4 %s\n',cfg.material.label);
    fprintf(fid,'modelType=%s\n',cfg.material.modelType);
    fprintf(fid,'GPU production path: large spectral/reconstruction/LWRQ calculations on GPU; gather only for control/I-O.\n');
    if strcmp(cfg.material.modelType,'mixed_fourier_B')
        fprintf(fid,['Medium B: c_B=%.8g; E_B/c_B = %.8g + %.8g cos(x1-x4) cos(x2-x5) ', ...
            '+ %.8g cos(x2-x5) cos(x3-x6) + %.8g cos(x3-x6) cos(x1-x4)\n'], ...
            cfg.material.cB,cfg.material.b0,cfg.material.a12,cfg.material.a34,cfg.material.a56);
        fprintf(fid,'Additional axial term inside brackets: %.8g * sum_l cos(x_l)\n',cfg.material.axialAmplitude);
    end
    fprintf(fid,'N=%d, NF=%d, reduced dimension=%d\n',cfg.problem.N,(2*cfg.problem.N)^6,2*(2*cfg.problem.N)^6);
    fprintf(fid,'Path: %d independent points, %d plotted points, %d modes\n', ...
        results.path.numIndependent,results.path.numPlot,cfg.spectrum.nev);
    fprintf(fid,'Continuum mean eps = %.12g; A-reference mean = %.12g; relative difference = %.3e\n', ...
        results.materialInfo.continuumMeanPermittivity,results.materialInfo.targetMeanPermittivity, ...
        results.materialInfo.relativeContinuumMeanMismatch);
    fprintf(fid,'Current invocation wall time = %.6f h\n',results.totalWallSeconds/3600);
    fprintf(fid,'Stored spectral q solve-time sum = %.6f h\n',sum(results.solverStats.wallTimeSeconds)/3600);
    fprintf(fid,'q checkpoints reused this invocation = %d/%d\n', ...
        nnz(results.solverStats.loadedFromCheckpoint),results.path.numIndependent);
    fprintf(fid,'Average stored q solve = %.3f s; mean outer = %.2f; mean Mhat CG = %.2f\n', ...
        mean(results.solverStats.wallTimeSeconds),mean(results.solverStats.outerSteps), ...
        mean(results.solverStats.mhatAverageIterations));
    fprintf(fid,'Max inverse Ritz residual = %.6e\n\n', ...
        max(results.solverStats.maxRelativeInverseRitzResidual));

    fprintf(fid,'Selected 3D LWRQ points:\n');
    for j=1:ns
        p=results.selected{j};
        fprintf(fid,'%s: point %d, mode %d, lambda6D=%.12e, lambdaLWRQ=%.12e, rel.diff=%.6e, eta3D=%.6e, nu=%.6e\n', ...
            p.label,p.pointIndex,p.modeIndex,p.lambda6D,p.lambdaLWRQ,p.deltaLambda,p.eta3D,p.nuLWRQ);
        fprintf(fid,'    q=[%.8f %.8f %.8f]^T; visual centers z0=%d, x0=%d, xy=%d; half-width=%.8f\n', ...
            p.q(1),p.q(2),p.q(3),p.visual.z0.numCenters,p.visual.x0.numCenters,p.visual.xy.numCenters,cfg.visual.halfWidth);
    end
end
