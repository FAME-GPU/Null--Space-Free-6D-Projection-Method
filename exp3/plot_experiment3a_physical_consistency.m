function fig = plot_experiment3a_physical_consistency(R)
%PLOT_EXPERIMENT3A_PHYSICAL_CONSISTENCY Manuscript Fig. 3(a) only.
%
% Main axes: worst-case eta_3D, nu_LWRQ and delta_lambda for the fixed
% half-width L=0.5 Yee-grid refinement. The four mesh cases are displayed
% at equal horizontal spacing; tick labels retain the physical mesh widths.
%
% Inset: complete h=0.025 window-sensitivity study for L=0.25,...,4.
% The inset is placed in the upper-right region with a visible right margin.
% Its scientific-notation multiplier is drawn manually so that it can be
% shifted left and kept clear of the title. The boxed legend is placed
% farther left to maintain visible separation from the inset.
%
% No subfigure label is drawn here--LaTeX supplies (a).
% Output geometry is intentionally identical to Fig. 3(b): 7.4 x 5.35 in.
% Both figures are written to <experiment-root>/figure/ as PNG + FIG only.

    if nargin < 1 || isempty(R)
        R = load_experiment3_results_local();
    end
    root = setup_experiment3_local();
    figureDir = fullfile(root,'figure');
    if ~exist(figureDir,'dir'), mkdir(figureDir); end
    validate_a(R);

    %% Mesh-refinement data.
    D = R.meshSummary;
    [h,ord] = sort([D.h],'descend');
    D = D(ord);
    eta = [D.maxEta3D];
    nu  = [D.maxNuLWRQ];
    dl  = [D.maxDeltaLambda];
    if numel(h) ~= 4
        error('Fig. 3(a) expects exactly four mesh-refinement cases.');
    end

    % Equal visual spacing for the four mesh cases.
    xMain = 1:numel(h);
    xLabels = arrayfun(@(x)sprintf('%g',x),h,'UniformOutput',false);

    %% Window-sensitivity inset data.
    W = R.windowSummary;
    [wp,ordW] = sort([W.halfWidth]);
    W = W(ordW);
    etaW = [W.maxEta3D];
    nuW  = [W.maxNuLWRQ];
    dlW  = [W.maxDeltaLambda];
    expectedW = [0.25 0.375 0.50 0.75 1.00 1.50 2.00 2.50 3.00 3.50 4.00];
    if numel(wp)~=numel(expectedW) || max(abs(wp-expectedW))>1e-12
        error('Fig. 3(a) expects the complete window half-width list from 0.25 to 4.');
    end

    %% Shared manuscript geometry: SAME as Fig. 3(b).
    fig = figure('Name','Experiment 3(a) - Physical-space consistency', ...
        'Color','w', ...
        'Units','inches', ...
        'Position',[0.8 0.8 7.4 5.35], ...
        'PaperPositionMode','auto');

    fontName = 'Times New Roman';
    fontSize = 17.0;
    lineWidth = 1.60;
    markerSize = 7.2;

    %% Main axes.
    ax = axes(fig,'Position',[0.105 0.125 0.845 0.825]);
    hold(ax,'on');
    p1 = plot(ax,xMain,eta,'o-','LineWidth',lineWidth,'MarkerSize',markerSize);
    p2 = plot(ax,xMain,nu, 's-','LineWidth',lineWidth,'MarkerSize',markerSize);
    p3 = plot(ax,xMain,dl, '^-','LineWidth',lineWidth,'MarkerSize',markerSize+0.4);
    set(ax, ...
        'FontName',fontName, ...
        'FontSize',fontSize, ...
        'LineWidth',0.9, ...
        'TickDir','in', ...
        'Layer','top', ...
        'XTick',xMain, ...
        'XTickLabel',xLabels);
    grid(ax,'on');
    box(ax,'on');
    xlim(ax,[0.80 numel(xMain)+0.20]);

    ymax = max([eta(:);nu(:);dl(:)]);
    ytop = ceil(1.02*ymax/1e-4)*1e-4;
    if ytop <= 0, ytop = 1.5e-3; end
    ylim(ax,[0 ytop]);
    ax.YAxis.Exponent = -3;

    xlabel(ax,'Yee mesh width $h$','Interpreter','latex','FontSize',21);
    ylabel(ax,'Worst-case relative error','Interpreter','latex','FontSize',21);

    %% Boxed common legend.
    % Fine tuning: [left bottom width height].
    % Shifted farther left than the previous version.
    lgd = legend(ax,[p1 p2 p3], ...
        {'$\max\eta_{3D}$','$\max\nu_{\mathrm{LWRQ}}$','$\max\delta_\lambda$'}, ...
        'Interpreter','latex', ...
        'Location','none', ...
        'Box','on', ...
        'FontSize',16.5);
    lgd.Units = 'normalized';
    lgd.Color = 'w';
    lgd.LineWidth = 0.9;
    lgd.Position = [0.315 0.700 0.205 0.185];

    %% Window-sensitivity inset.
    % Fine tuning: [left bottom width height].
    axInset = axes(fig,'Position',[0.615 0.515 0.310 0.315]);
    hold(axInset,'on');

    % Scale the inset ordinate by 1e4 explicitly. This keeps the displayed
    % tick labels compact while allowing the x10^{-4} multiplier to be
    % positioned manually, avoiding overlap with the inset title.
    insetScale = 1e4;
    plot(axInset,wp,insetScale*etaW,'o-','LineWidth',1.25,'MarkerSize',5.2);
    plot(axInset,wp,insetScale*nuW, 's-','LineWidth',1.25,'MarkerSize',5.2);
    plot(axInset,wp,insetScale*dlW, '^-','LineWidth',1.25,'MarkerSize',5.5);
    set(axInset, ...
        'FontName',fontName, ...
        'FontSize',13.5, ...
        'LineWidth',0.8, ...
        'TickDir','in', ...
        'Layer','top');
    grid(axInset,'on');
    box(axInset,'on');
    xlim(axInset,[0.18 4.08]);
    xticks(axInset,[0.25 0.5 1 2 3 4]);
    xticklabels(axInset,{'0.25','0.5','1','2','3','4'});

    ymaxWScaled = insetScale*max([etaW(:);nuW(:);dlW(:)]);
    ytopWScaled = ceil(1.05*ymaxWScaled*10)/10;
    if ytopWScaled <= 0, ytopWScaled = 1.2; end
    ylim(axInset,[0 ytopWScaled]);
    axInset.YAxis.Exponent = 0;

    xlabel(axInset,'window half-width $L=nh$','Interpreter','latex','FontSize',15.5);
    ylabel(axInset,'relative','Interpreter','latex','FontSize',15.5);
    title(axInset,'window sensitivity, $h=0.025$', ...
        'Interpreter','latex','FontSize',15.0,'FontWeight','normal');

    % Manual scientific-notation multiplier. Adjust the first coordinate
    % to move it farther left/right without disturbing the title.
    text(axInset,-0.025,1.025,'$\times10^{-4}$', ...
        'Units','normalized', ...
        'Interpreter','latex', ...
        'FontName',fontName, ...
        'FontSize',13.5, ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','bottom', ...
        'Clipping','off');

    %% Save in the shared manuscript figure directory.
    basePath = fullfile(figureDir,'Figure_Exp3a_PhysicalConsistency');
    save_experiment3_png_fig(fig,basePath,400);
    fprintf('Saved Fig. 3(a):\n  %s.png\n  %s.fig\n',basePath,basePath);
end

function validate_a(R)
    req = {'meshSummary','windowSummary'};
    for k = 1:numel(req)
        if ~isfield(R,req{k}) || isempty(R.(req{k}))
            error('Missing results.%s required for Fig. 3(a).',req{k});
        end
    end
end
