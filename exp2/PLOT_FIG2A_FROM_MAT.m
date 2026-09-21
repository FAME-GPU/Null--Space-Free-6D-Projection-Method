function PLOT_FIG2A_FROM_MAT()
%PLOT_FIG2A_FROM_MAT Replot Fig. 2(a) without rerunning the GPU experiment.
    rootDir = setup_experiment2_package();
    cfg = make_config_experiment2_allgpu();
    p = fullfile(rootDir,'output',cfg.output.resultFile);
    if exist(p,'file')~=2, error('Result MAT file not found: %s',p); end
    S = load(p,'results');
    check_experiment2_results(S.results,cfg);
    plot_experiment2a_tradeoff(S.results,cfg,rootDir);
end
