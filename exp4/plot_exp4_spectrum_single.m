function fig = plot_exp4_spectrum_single(R,saveFigure)
    root=setup_exp4(); mediumName=R.cfg.material.name;
    vals=cellfun(@(p)p.lambdaLWRQ,R.selected);
    yLim=[0 1.24*max([R.spectrum.lambdaPlot(:); vals(:)])];
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

