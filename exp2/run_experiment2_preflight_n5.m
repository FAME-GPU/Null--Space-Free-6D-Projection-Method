function run_experiment2_preflight_n5(cfg,rootDir)
%RUN_EXPERIMENT2_PREFLIGHT_N5 Exp2-named wrapper for legacy N=5 preflight.
    if nargin<2, rootDir=[]; end
    run_experiment3_preflight_n5(cfg,rootDir);
end
