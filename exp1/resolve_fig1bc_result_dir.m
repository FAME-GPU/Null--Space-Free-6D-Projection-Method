function dataDir = resolve_fig1bc_result_dir()
%RESOLVE_FIG1BC_RESULT_DIR Public package uses one canonical results directory.
    cfg = make_config_exp1bc();
    dataDir = fullfile(exp1bc_project_root(),cfg.output.resultsDirectory);
end
