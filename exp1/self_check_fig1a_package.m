function self_check_fig1a_package()
%SELF_CHECK_FIG1A_PACKAGE Lightweight static/package checks.
    root=fileparts(which('START_SELF_CHECK')); if isempty(root),root=pwd;end
    req={ ...
      'make_config_A_fig1a','run_method_A_fig1a','fig1a_make_A_eigs_top_preconditioned', ...
      'make_config_BC_fig1a','run_fig1a_BC_method','run_method_B_fig1','run_method_C_fig1', ...
      'Lanczos','fig1a_build_mass_backend_large_scalar','plot_fig1a_final','check_fig1a_results'};
    for i=1:numel(req)
        if isempty(which(req{i})), error('Missing function on path: %s',req{i}); end
    end
    cfgA=make_config_A_fig1a();
    assert(cfgA.problem.N==5 && cfgA.shift.minresTolerance==1e-10);
    assert(cfgA.shift.plateauAcceptanceCeiling==1e-9);
    cfgBC=make_config_BC_fig1a();
    assert(cfgBC.problem.N==5 && cfgBC.lanczos.krylovDimension==40);
    assert(strcmp(cfgA.mass.backend,cfgBC.mass.backend));
    assert(strcmp(cfgA.output.directory,'results') && strcmp(cfgBC.output.directory,'results'));
    fprintf('Exp1a package self-check passed.\n');
end
