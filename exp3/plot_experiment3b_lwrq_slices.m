function fig = plot_experiment3b_lwrq_slices(R)
%PLOT_EXPERIMENT3B_LWRQ_SLICES Manuscript Fig. 3(b) only.
%
% Relative local LWRQ deviation on the maximum L=4 window for tracked
% modes 11 and 114, on the Cartesian planes z=0 and x=0. All four maps
% share one logarithmic color scale.
%
% For the 2-by-2 block, only OUTER tick labels and coordinate labels are
% retained, following standard compact multi-panel figure practice:
%   - top row: x tick labels omitted;
%   - right column: y tick labels omitted;
%   - x-axis labels appear only on the bottom row;
%   - y-axis labels appear only on the left column.
% Tick marks themselves remain on all axes.
%
% No subfigure label is drawn here--LaTeX supplies (b).
% Output geometry is intentionally identical to Fig. 3(a): 7.4 x 5.35 in.
% Both figures are written to <experiment-root>/figure/ as PNG + FIG only.

    if nargin < 1 || isempty(R)
        R = load_experiment3_results_local();
    end
    root = setup_experiment3_local();
    figureDir = fullfile(root,'figure');
    if ~exist(figureDir,'dir'), mkdir(figureDir); end
    validate_b(R);

    modes = R.eigen.modes(:).';
    Z = cell(2,2);
    coords = cell(2,1);
    for j = 1:2
        S = R.spatialMaxWindow{j};
        if isempty(S), error('Missing results.spatialMaxWindow{%d}.',j); end
        reqS = {'coord','relativeDeviation_z0','relativeDeviation_x0','h','n','halfWidth'};
        for q = 1:numel(reqS)
            if ~isfield(S,reqS{q})
                error('Missing maximum-window spatial field %s.',reqS{q});
            end
        end
        if S.n~=160 || abs(S.h-0.025)>1e-14 || abs(S.halfWidth-4)>1e-12
            error('Fig. 3(b) spatial data must be the h=0.025, n=160, L=4 case.');
        end
        coords{j} = S.coord(:).';
        Z{j,1} = double(S.relativeDeviation_z0);
        Z{j,2} = double(S.relativeDeviation_x0);
    end

    %% Common logarithmic color range.
    zmin = inf;
    zmax = 0;
    for j = 1:2
        for k = 1:2
            z = Z{j,k};
            if any(~isfinite(z(:)))
                error('Nonfinite LWRQ deviation for mode %d.',modes(j));
            end
            zp = z(z>0);
            if ~isempty(zp), zmin = min(zmin,min(zp(:))); end
            if ~isempty(z), zmax = max(zmax,max(z(:))); end
        end
    end
    if ~isfinite(zmin) || ~(zmax>0)
        error('Spatial LWRQ deviations must contain positive values.');
    end
    cmin = 10^floor(log10(zmin));
    cmax = 10^ceil(log10(zmax));
    if cmax<=cmin, cmax=10*cmin; end

    %% Shared manuscript geometry: SAME as Fig. 3(a).
    fig = figure('Name','Experiment 3(b) - Maximum-window LWRQ deviation', ...
        'Color','w', ...
        'Units','inches', ...
        'Position',[0.8 0.8 7.4 5.35], ...
        'PaperPositionMode','auto');
    fontName = 'Times New Roman';

    % Manual 2x2 layout optimized for the 7.4 x 5.35 canvas.
    xLeft   = 0.090;
    xRight  = 0.490;
    wAx     = 0.325;
    yBottom = 0.110;
    yTop    = 0.555;
    hAx     = 0.355;

    ax = gobjects(2,2);
    for j = 1:2
        if j==1, yPos=yTop; else, yPos=yBottom; end
        for k = 1:2
            if k==1, xPos=xLeft; else, xPos=xRight; end
            ax(j,k) = axes(fig,'Position',[xPos yPos wAx hAx]);
            zplot = max(Z{j,k},cmin);
            imagesc(ax(j,k),coords{j},coords{j},zplot.');
            axis(ax(j,k),'image');
            axis(ax(j,k),'xy');
            set(ax(j,k), ...
                'FontName',fontName, ...
                'FontSize',16.5, ...
                'LineWidth',0.85, ...
                'TickDir','in', ...
                'ColorScale','log');
            caxis(ax(j,k),[cmin cmax]);
            box(ax(j,k),'on');
            xticks(ax(j,k),[-4 -2 0 2 4]);
            yticks(ax(j,k),[-4 -2 0 2 4]);

            if k==1
                planeText='$z=0$';
            else
                planeText='$x=0$';
            end
            title(ax(j,k),sprintf('mode %d, %s',modes(j),planeText), ...
                'Interpreter','latex','FontSize',18.0,'FontWeight','normal');

            % Keep only OUTER coordinate labels/tick labels.
            % Top row: suppress x tick labels and x-axis labels.
            if j==1
                ax(j,k).XTickLabel = [];
                xlabel(ax(j,k),'');
            else
                if k==1
                    xlabel(ax(j,k),'$x$','Interpreter','latex','FontSize',18.5);
                else
                    xlabel(ax(j,k),'$y$','Interpreter','latex','FontSize',18.5);
                end
            end

            % Right column: suppress y tick labels and y-axis labels.
            if k==1
                ylabel(ax(j,k),'$y$','Interpreter','latex','FontSize',18.5);
            else
                ax(j,k).YTickLabel = [];
                ylabel(ax(j,k),'');
            end
        end
    end

    cb = colorbar(ax(1,2),'Position',[0.905 0.165 0.020 0.690]);
    cb.FontName = fontName;
    cb.FontSize = 16.0;
    cb.Label.String = '$d_\zeta=|\rho_\zeta-\lambda_{3D}|/|\lambda_{3D}|$';
    cb.Label.Interpreter = 'latex';
    cb.Label.FontSize = 18.0;

    %% Save in exactly the same directory as Fig. 3(a).
    basePath = fullfile(figureDir,'Figure_Exp3b_LWRQDeviationSlices');
    save_experiment3_png_fig(fig,basePath,400);
    fprintf('Saved Fig. 3(b):\n  %s.png\n  %s.fig\n',basePath,basePath);
end

function validate_b(R)
    req = {'eigen','spatialMaxWindow'};
    for k = 1:numel(req)
        if ~isfield(R,req{k})
            error('Missing results.%s required for Fig. 3(b).',req{k});
        end
    end
    if ~isfield(R.eigen,'modes') || numel(R.eigen.modes)~=2 || numel(R.spatialMaxWindow)~=2
        error('Fig. 3(b) expects exactly two tracked modes and two spatial data sets.');
    end
end
