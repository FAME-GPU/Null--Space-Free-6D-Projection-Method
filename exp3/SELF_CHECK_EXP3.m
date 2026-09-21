function SELF_CHECK_EXP3()
%SELF_CHECK_EXP3 Static audit of the final Experiment 3 package.
    root=setup_experiment3_local();
    cfg=make_config_experiment4();
    names={'run_experiment4','write_experiment4_outputs','get_selected_eigenpairs_exp4', ...
           'compute_yee_lwrq_metrics_exp4','compute_lwrq_exp4','prepare_taylor_fourier_basis_exp4', ...
           'reconstruct_single_mode_large_window_exp3','check_large_window_gpu_capacity_exp3', ...
           'plot_experiment3a_physical_consistency','plot_experiment3b_lwrq_slices'};
    for k=1:numel(names)
        p=which(names{k},'-all');
        if isempty(p), error('Missing function: %s',names{k}); end
        if iscell(p) && numel(p)~=1, error('Duplicate function on path: %s',names{k}); end
    end
    NF=(2*cfg.problem.N)^cfg.problem.dim;
    assert(NF==1e6);
    assert(isequal(cfg.tracking.n5Modes,[11 114]));
    assert(cfg.window.masterN==160 && abs(cfg.window.h-0.025)<1e-14);
    hw=cfg.window.h*cfg.window.n;
    expected=[0.25 0.375 0.5 0.75 1 1.5 2 2.5 3 3.5 4];
    assert(numel(hw)==numel(expected) && max(abs(hw-expected))<1e-14);
    metrics=fileread(which('compute_yee_lwrq_metrics_exp4'));
    assert(contains(metrics,'relativeDeviation_x0') && ~contains(metrics,'relativeDeviation_xy'));
    pa=fileread(which('plot_experiment3a_physical_consistency'));
    pb=fileread(which('plot_experiment3b_lwrq_slices'));
    geom = "'Position',[0.8 0.8 7.4 5.35]";
    assert(contains(pa,geom) && contains(pb,geom));
    assert(contains(pb,'relativeDeviation_x0') && ~contains(pb,'relativeDeviation_xy'));
    fprintf('[Experiment 3 final package static audit] PASS\n');
    fprintf('  root: %s\n',root);
    fprintf('  N=%d, NF=%d, modes=%s\n',cfg.problem.N,NF,mat2str(cfg.tracking.n5Modes));
    fprintf('  Fig. 3 data: mesh refinement + L=0.25...4 window study + z=0/x=0 slices\n');
end
