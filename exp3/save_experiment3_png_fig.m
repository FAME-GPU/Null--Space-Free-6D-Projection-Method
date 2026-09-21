function save_experiment3_png_fig(fig,basePath,resolution)
%SAVE_EXPERIMENT3_PNG_FIG Save a paper figure as PNG plus editable MATLAB FIG.
% PDF/EPS output is intentionally not produced in this project.

    if nargin < 3 || isempty(resolution), resolution = 400; end
    [folder,~,~] = fileparts(basePath);
    if ~isempty(folder) && ~exist(folder,'dir'), mkdir(folder); end

    drawnow;
    exportgraphics(fig,[basePath '.png'],'Resolution',resolution,'BackgroundColor','white');
    savefig(fig,[basePath '.fig']);
end
