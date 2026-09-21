function results = RUN_EXPERIMENT2_FIG2A()
%RUN_EXPERIMENT2_FIG2A Complete production entry for manuscript Fig. 2(a).
% Solves the fixed N=5 problem, constructs the n=40 direct reference,
% performs the p/center Taylor trade-off, validates the result, and exports
% the final Fig. 2(a) as EPS/PNG/FIG.

    rootDir = setup_experiment2_package();
    cfg = make_config_experiment2_allgpu();
    cfg.run.makePlots = false; % plot only after result validation

    results = run_experiment2_allgpu(cfg,rootDir);
    check_experiment2_results(results,cfg);
    plot_experiment2a_tradeoff(results,cfg,rootDir);

    fprintf('\nFig. 2(a) complete.\n');
    fprintf('  data   : %s\n',fullfile(rootDir,'output',cfg.output.resultFile));
    fprintf('  figure : %s\n',fullfile(rootDir,'figures',[cfg.output.figureTradeoffBase '.png']));
end
