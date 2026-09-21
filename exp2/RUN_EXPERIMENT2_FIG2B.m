function results = RUN_EXPERIMENT2_FIG2B()
%RUN_EXPERIMENT2_FIG2B Complete production entry for manuscript Fig. 2(b).
% Solves the fixed N=5 problem once, performs the memory-safe n-scaling
% reconstruction experiment, validates the result, and exports the final
% Fig. 2(b) as EPS/PNG/FIG.

    rootDir = setup_experiment2_package();
    cfg = make_config_experiment2b_ng_scaling();
    cfg.run.makePlot = false; % plot only after result validation

    results = run_experiment2b_ng_scaling(cfg,rootDir);
    check_experiment2b_results(results,cfg,true);
    plot_experiment2b_ng_scaling(results,cfg,rootDir);

    fprintf('\nFig. 2(b) complete.\n');
    fprintf('  data   : %s\n',fullfile(rootDir,'output',cfg.output.resultFile));
    fprintf('  CSV    : %s\n',fullfile(rootDir,'output',cfg.output.scalingCSV));
    fprintf('  figure : %s\n',fullfile(rootDir,'figures',[cfg.output.figureBase '.png']));
end
