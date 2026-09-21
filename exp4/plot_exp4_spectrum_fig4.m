function figs = plot_exp4_spectrum_fig4(A,B,saveFigure)
%PLOT_EXP4_SPECTRUM_FIG4 Generate the two spectrum panels used in Figure 4.
%   Dark-blue curves are the 6D eigenvalues. Hollow red markers are the
%   LWRQ values computed from the reconstructed 3D Yee fields at S1--S4.

    if nargin<3, saveFigure=false; end
    root = setup_exp4();
    validate_pair(A,B);

    yMax = max([A.spectrum.lambdaPlot(:); B.spectrum.lambdaPlot(:); ...
                selected_lwrq_values(A); selected_lwrq_values(B)]);
    if ~(isfinite(yMax) && yMax>0)
        error('Invalid spectrum range.');
    end
    yLim = [0, 1.24*yMax];

    figs = struct();
    figs.A = plot_one_medium(A,'A',yLim,saveFigure,root);
    figs.B = plot_one_medium(B,'B',yLim,saveFigure,root);
end

function fig = plot_one_medium(R,mediumName,yLim,saveFigure,root)
    bandColor = [0.04 0.27 0.52];
    recoveryColor = [0.80 0.10 0.10];
    boundaryColor = [0.63 0.63 0.63];

    fontName = 'Times New Roman';
    fsTick   = 9.2;
    fsLabel  = 10.0;
    fsLegend = 7.6;

    lwBand         = 0.78;
    markerRecovery = 4.7;
    lwRecovery     = 1.00;

    % Slightly larger canvas and larger bottom/left margins so the x/y axis
    % labels are not clipped after export and LaTeX insertion.
    fig = figure('Name',sprintf('Medium %s spectrum',mediumName), ...
        'Units','inches','Position',[0.5 0.5 2.42 2.26],'Color','w');

    ax = axes(fig,'Position',[0.17 0.19 0.79 0.72]);
    hold(ax,'on'); box(ax,'on');

    L = R.spectrum.lambdaPlot;   % nev-by-numPlot
    x = R.path.plotIndex(:).';

    for m = 1:size(L,1)
        plot(ax,x,L(m,:),'-','Color',bandColor,'LineWidth',lwBand, ...
            'HandleVisibility','off');
    end

    % Path-segment boundaries.
    b = R.path.segmentBoundaries;
    for k = 2:numel(b)-1
        xline(ax,b(k),'--','Color',boundaryColor,'LineWidth',0.65, ...
            'HandleVisibility','off');
    end

    % LWRQ recovered values: hollow red markers.
    markerList = {'o','s','d','^'};
    legendHandles = gobjects(1,4);
    for j = 1:4
        p = R.selected{j};
        legendHandles(j) = plot(ax,p.pointIndex,p.lambdaLWRQ,markerList{j}, ...
            'LineStyle','none', ...
            'MarkerSize',markerRecovery, ...
            'MarkerFaceColor','none', ...
            'MarkerEdgeColor',recoveryColor, ...
            'Color',recoveryColor, ...
            'LineWidth',lwRecovery, ...
            'DisplayName',sprintf('S%d',j));
    end

    endpointTicks = b;
    endpointLabels = {'$q^{(0)}$','$q^{(1)}$','$q^{(2)}$','$q^{(3)}$','$q^{(0)}$'};

    set(ax,'XLim',[1 R.path.numPlot],'YLim',yLim, ...
        'XTick',endpointTicks,'XTickLabel',endpointLabels, ...
        'TickLabelInterpreter','latex', ...
        'FontName',fontName,'FontSize',fsTick, ...
        'TickDir','out','LineWidth',0.8,'Layer','top');

    xl = xlabel(ax,'Bloch path','FontName',fontName,'FontSize',fsLabel);
    yl = ylabel(ax,'$\lambda$','Interpreter','latex', ...
        'FontName',fontName,'FontSize',fsLabel);

    % Move the labels slightly inward to reduce clipping risk.
    xl.Units = 'normalized';
    yl.Units = 'normalized';
    xl.Position(2) = -0.12;
    yl.Position(1) = -0.12;

    grid(ax,'on');
    ax.GridAlpha = 0.10;
    ax.MinorGridAlpha = 0.06;
    ax.YMinorTick = 'off';

    % One-row legend placed high in the reserved headroom region.
    lg = legend(ax,legendHandles,{'$S_1$','$S_2$','$S_3$','$S_4$'}, ...
        'Interpreter','latex', ...
        'Location','none', ...
        'NumColumns',4, ...
        'FontName',fontName,'FontSize',fsLegend, ...
        'Box','on');
    lg.ItemTokenSize = [8 7];
    lg.Units = 'normalized';
    lg.Position = [0.19 0.78 0.67 0.10];

    if saveFigure
        base = sprintf('Figure_Exp4_Medium%s_Spectrum_Line_N5',mediumName);
        exportgraphics(fig,fullfile(root,'figure',[base '.png']),'Resolution',600);
        exportgraphics(fig,fullfile(root,'figure',[base '.pdf']),'ContentType','vector');
        savefig(fig,fullfile(root,'figure',[base '.fig']));
    end
end

function v = selected_lwrq_values(R)
    v = zeros(numel(R.selected),1);
    for j = 1:numel(R.selected)
        v(j) = R.selected{j}.lambdaLWRQ;
    end
end

function validate_pair(A,B)
    if ~isequal(size(A.spectrum.lambdaPlot),size(B.spectrum.lambdaPlot))
        error('Medium A/B spectrum sizes differ.');
    end
    if A.path.numPlot ~= B.path.numPlot
        error('Medium A/B path lengths differ.');
    end
    if numel(A.selected)~=4 || numel(B.selected)~=4
        error('Expected exactly four selected recovery points S1--S4 in both media.');
    end
end