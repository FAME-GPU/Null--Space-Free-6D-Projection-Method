function SELF_CHECK_EXPERIMENT2_PACKAGE()
%SELF_CHECK_EXPERIMENT2_PACKAGE Static consistency checks; no GPU solve.
    rootDir = setup_experiment2_package();
    required = { ...
        'RUN_EXPERIMENT2_FIG2A','RUN_EXPERIMENT2_FIG2B', ...
        'make_config_experiment2_allgpu','make_config_experiment2b_ng_scaling', ...
        'run_experiment2_allgpu','run_experiment2b_ng_scaling', ...
        'plot_experiment2a_tradeoff','plot_experiment2b_ng_scaling', ...
        'build_experiment3_system','solve_modes_exp3','gpu_initialize'};
    for k=1:numel(required)
        p=which(required{k});
        if isempty(p), error('Missing required function: %s',required{k}); end
        if ~startsWith(p,rootDir), error('Function shadowing detected: %s -> %s',required{k},p); end
    end

    a=make_config_experiment2_allgpu();
    b=make_config_experiment2b_ng_scaling();
    assert(a.problem.N==5 && b.problem.N==5);
    assert(isequal(a.problem.P,b.problem.P));
    assert(isequal(a.problem.q,b.problem.q));
    assert(isequal(a.problem.T,b.problem.T));
    assert(a.material.epsC==b.material.epsC && a.material.alpha==b.material.alpha);
    assert(a.mass.rank1.N5.M==b.mass.rank1.N5.M);
    assert(isequal(a.mass.rank1.N5.z,b.mass.rank1.N5.z));
    assert(a.spectrum.performanceMode==10 && b.spectrum.performanceMode==10);
    assert(a.yee.n==b.scaling.baselineN);
    assert(abs(a.yee.h-b.scaling.h)<100*eps);
    assert(a.taylor.performanceOrder==b.scaling.taylorOrder);
    assert(a.taylor.blocksPerDimension==b.scaling.baselineBlocksPerDimension);
    assert(b.scaling.timeoutSeconds==1e4);
    assert(isequal(b.scaling.nValues,[40 60 80 100 120 140 160 180 200]));

    fprintf('Experiment 2 package self-check PASSED.\n');
    fprintf('  Fig. 2(a): N=5, n=40, p={6,8,10}, centers=1^3,...,6^3.\n');
    fprintf('  Fig. 2(b): n=40:20:200, p=10, radius-controlled centers, 1e4-s budgets.\n');
end
