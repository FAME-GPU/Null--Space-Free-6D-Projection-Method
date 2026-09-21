function [figA,figB] = PLOT_EXP3()
%PLOT_EXP3 Generate the two final manuscript Fig. 3 images from output/.
%   The two figures are saved independently in <root>/figure/ with the
%   same 7.4 x 5.35 inch outer geometry. LaTeX combines them as subfigures.

    root = setup_experiment3_local();
    R = load_experiment3_results_local();
    clean_experiment3_figure_outputs(root);
    figA = plot_experiment3a_physical_consistency(R);
    figB = plot_experiment3b_lwrq_slices(R);
    extract_experiment3_paper_data(R);
    fprintf('Experiment 3 Fig. 3(a)/3(b) generated in: %s\n',fullfile(root,'figure'));
end
