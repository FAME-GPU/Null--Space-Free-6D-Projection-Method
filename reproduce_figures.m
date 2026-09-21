function reproduce_figures()
%REPRODUCE_FIGURES Plot all four experiments from locally generated MAT files.
% Run run_experiment(1), ..., run_experiment(4) before using this entry.
    for id=1:4, plot_experiment(id); end
end
