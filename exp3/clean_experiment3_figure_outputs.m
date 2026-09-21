function clean_experiment3_figure_outputs(root)
%CLEAN_EXPERIMENT3_FIGURE_OUTPUTS Remove package-generated figure files only.
% This prevents legacy diagnostic PDFs/FIGs from being mistaken for the
% manuscript-facing Fig. 3 after rerunning START_LOCAL_PLOT_ALL.

    if nargin < 1 || isempty(root)
        root = setup_experiment3_local();
    end
    figDir = fullfile(root,'figure');
    if ~exist(figDir,'dir'), mkdir(figDir); end

    patterns = { ...
        'Experiment3_*.pdf', ...       % legacy vector exports
        'Experiment3_*.png', ...       % optional diagnostic PNG exports
        'Experiment3_*.fig', ...       % optional diagnostic FIG exports
        'Figure_Exp3*.png', ...        % old/new manuscript-facing PNG
        'Figure_Exp3*.fig'};           % old/new manuscript-facing editable FIG

    for k = 1:numel(patterns)
        D = dir(fullfile(figDir,patterns{k}));
        for j = 1:numel(D)
            delete(fullfile(D(j).folder,D(j).name));
        end
    end
end
