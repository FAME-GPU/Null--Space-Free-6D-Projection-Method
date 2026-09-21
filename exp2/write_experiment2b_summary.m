function write_experiment2b_summary(results,cfg,rootDir)
%WRITE_EXPERIMENT2B_SUMMARY Human-readable scaling summary.
    if nargin<3 || isempty(rootDir), rootDir=fileparts(mfilename('fullpath')); end
    outDir=fullfile(rootDir,'output');
    path=fullfile(outDir,cfg.output.summaryTXT);
    fid=fopen(path,'w');
    if fid<0, error('write_experiment2b_summary:OpenFailed','Cannot open summary file.'); end
    cleaner=onCleanup(@()fclose(fid)); %#ok<NASGU>

    T=results.scaling;
    fprintf(fid,'Experiment 2(b): N_G scaling of Yee-field reconstruction\n');
    fprintf(fid,'N=%d, NF=%d, mode=%d, lambda=%.16e\n',results.N,results.NF,results.mode,results.lambdaMode);
    fprintf(fid,'h=%.16g, p=%d\n',cfg.scaling.h,cfg.scaling.taylorOrder);
    fprintf(fid,'exact adopted r_ph,max threshold = %.16e (rounded %.2f)\n',results.rhoTarget,results.rhoTarget);
    fprintf(fid,'final/current direct point batch = %d (per-row values are stored in CSV)\n',results.directBlockSize);
    fprintf(fid,'final/current Taylor term batch = %d (per-row values are stored in CSV)\n',results.termBatchSize);
    fprintf(fid,'independent per-method budget = %.1f s\n',results.timeoutSeconds);
    fprintf(fid,'No full N_G coordinate or field arrays are stored; reconstruction is streamed by local blocks.\n\n');

    fprintf(fid,'%5s %12s %4s %9s %12s %14s %14s %13s %13s\n', ...
        'n','N_G','k','centers','r_ph,max','direct(s)','Taylor(s)','e2','eInf');
    for i=1:height(T)
        fprintf(fid,'%5d %12d %4d %9d %12.6f %14s %14s %13s %13s\n', ...
            T.n(i),T.Ng(i),T.BlocksPerDimension(i),T.NumCenters(i),T.RhoMax(i), ...
            fmt(T.DirectTimeSeconds(i),T.DirectStatus(i)), ...
            fmt(T.TaylorTimeSeconds(i),T.TaylorStatus(i)), ...
            fmt_num(T.RelativeFieldError2(i)),fmt_num(T.RelativeFieldErrorInf(i)));
    end
end

function s=fmt(v,status)
    if isfinite(v), s=sprintf('%.4f',v); else, s=char(status); end
end
function s=fmt_num(v)
    if isfinite(v), s=sprintf('%.6e',v); else, s='--'; end
end
