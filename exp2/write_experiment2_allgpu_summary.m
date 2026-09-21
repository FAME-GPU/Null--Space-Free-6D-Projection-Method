function write_experiment2_allgpu_summary(results,cfg,rootDir)
%WRITE_EXPERIMENT2_ALLGPU_SUMMARY Manuscript-facing all-GPU values.
    outDir=fullfile(rootDir,'output');
    fid=fopen(fullfile(outDir,cfg.output.summaryTXT),'w');
    if fid<0, warning('Could not open summary file.'); return; end
    c=onCleanup(@()fclose(fid)); %#ok<NASGU>
    fprintf(fid,'Experiment 2: N=5 all-GPU 3-D Yee-field reconstruction\n\n');
    fprintf(fid,'GPU=%s\n',results.gpu.name);
    fprintf(fid,'N=%d, NF=%d, reduced dimension=%d\n',results.N5.N,results.N5.NF,results.N5.reducedDimension);
    fprintf(fid,'Yee n=%d, h=%.8f, Ng=%d\n\n',results.reference.n,results.reference.h,results.reference.Ng);
    fprintf(fid,'Adopted Taylor: p=%d, centers=%d, mode=%d\n',results.adopted.order,results.adopted.numCenters,results.adopted.mode);
    fprintf(fid,'r_ph,max single/adopted = %.8e / %.8e\n',results.adopted.rhoSingle,results.adopted.rhoAdopted);
    fprintf(fid,'adopted e2/eA = %.8e / %.8e\n\n',results.adopted.relativeFieldError,results.adopted.curlCurlActionError);
    T=results.tradeoff;
    fprintf(fid,'Figure 2(a): GPU order-center trade-off\n');
    for i=1:height(T)
        fprintf(fid,'p=%2d centers=%3d rph=%.6e e2=%.6e GPU=%.6e s peakAux=%.3f MiB\n', ...
            T.p(i),T.NumCenters(i),T.RhoMax(i),T.RelativeFieldErrorMode10(i),T.GPUTimeSeconds(i),T.EstimatedPeakAuxMemoryMiB(i));
    end
    W=results.workflow;
    fprintf(fid,'\nAuxiliary workflow timing diagnostic (not final Figure 2(b))\n');
    for i=1:height(W)
        fprintf(fid,'%s: eig %.6e s, recon %.6e s, other %.6e s, total %.6e s\n', ...
            W.Workflow{i},W.EigensolverSeconds(i),W.ReconstructionSeconds(i),W.OtherPreparationSeconds(i),W.GPUWorkflowTotalSeconds(i));
    end
    fprintf(fid,'direct/Taylor GPU numerical total ratio = %.8f\n',results.workflowRatio);
    fprintf(fid,'host/setup-inclusive totals = %.8e / %.8e s\n',W.FullWorkflowTotalSeconds(1),W.FullWorkflowTotalSeconds(2));

    fl=fopen(fullfile(outDir,cfg.output.latexTXT),'w');
    if fl>=0
        cc=onCleanup(@()fclose(fl)); %#ok<NASGU>
        fprintf(fl,'N = %d\nNF = %d\nNg = %d\n',results.N5.N,results.N5.NF,results.reference.Ng);
        fprintf(fl,'workflow_direct_eigensolver = %.8e\nworkflow_direct_reconstruction = %.8e\nworkflow_direct_other = %.8e\nworkflow_direct_total = %.8e\n', ...
            W.EigensolverSeconds(1),W.ReconstructionSeconds(1),W.OtherPreparationSeconds(1),W.GPUWorkflowTotalSeconds(1));
        fprintf(fl,'workflow_taylor_eigensolver = %.8e\nworkflow_taylor_reconstruction = %.8e\nworkflow_taylor_other = %.8e\nworkflow_taylor_total = %.8e\n', ...
            W.EigensolverSeconds(2),W.ReconstructionSeconds(2),W.OtherPreparationSeconds(2),W.GPUWorkflowTotalSeconds(2));
        fprintf(fl,'workflow_ratio = %.8f\nadopted_mode10_e2 = %.8e\nadopted_mode10_eA = %.8e\nrph_single = %.8e\nrph_multi = %.8e\n', ...
            results.workflowRatio,results.adopted.relativeFieldError,results.adopted.curlCurlActionError,results.adopted.rhoSingle,results.adopted.rhoAdopted);
    end
end
