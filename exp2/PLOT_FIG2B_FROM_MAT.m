function PLOT_FIG2B_FROM_MAT()
%PLOT_FIG2B_FROM_MAT Replot Fig. 2(b) without rerunning the GPU experiment.
    rootDir = setup_experiment2_package();
    cfg = make_config_experiment2b_ng_scaling();
    p = fullfile(rootDir,'output',cfg.output.resultFile);
    if exist(p,'file')~=2, error('Result MAT file not found: %s',p); end
    S = load(p,'results');
    cfg.scaling.nValues = S.results.scaling.n(:).';
    check_experiment2b_results(S.results,cfg,true);
    plot_experiment2b_ng_scaling(S.results,cfg,rootDir);
end
