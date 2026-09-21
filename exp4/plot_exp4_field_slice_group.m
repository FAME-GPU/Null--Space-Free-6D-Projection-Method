function fig = plot_exp4_field_slice_group(R,sliceKind,saveFigure)
%PLOT_EXP4_FIELD_SLICE_GROUP Plot S1--S4 on one common physical slice.
%
%   FIG = PLOT_EXP4_FIELD_SLICE_GROUP(R,'z0',SAVEFIGURE) creates a 2-by-2
%   group containing the z=0 normalized |E|^2 slices for S1--S4.
%
%   FIG = PLOT_EXP4_FIELD_SLICE_GROUP(R,'xy',SAVEFIGURE) creates the same
%   2-by-2 group on x=y.
%
%   Each selected mode was normalized jointly over its three reconstructed
%   visualization planes during the terminal postprocessing. Consequently,
%   every tile uses the common displayed range [0,1], while the four modes
%   are compared here by spatial pattern rather than absolute amplitude.

    if nargin<3, saveFigure=false; end
    root = setup_exp4();

    sliceKind = lower(char(sliceKind));
    if ~ismember(sliceKind,{'z0','xy'})
        error('sliceKind must be ''z0'' or ''xy''.');
    end
    if numel(R.selected)~=4
        error('Expected exactly four selected modes S1--S4.');
    end

    switch sliceKind
        case 'z0'
            sliceLabel = 'z=0';
            xLabel = '$x$';
            yLabel = '$y$';
            fileTag = 'z0';

        case 'xy'
            sliceLabel = 'x=y';
            xLabel = '$s$';
            yLabel = '$z$';
            fileTag = 'xy';
    end

    fontName = 'Times New Roman';
    fsTick    = 8.2;
    fsTitle   = 9.0;
    fsAxis    = 9.5;
    fsColorbar = 7.9;

    % Slightly wider figure and much more efficient internal layout.
    fig = figure('Name',sprintf('%s %s S1-S4 field group',R.cfg.output.tag,sliceLabel), ...
        'Units','inches','Position',[0.5 0.5 2.46 2.16],'Color','w');

    tl = tiledlayout(fig,2,2,'TileSpacing','compact','Padding','compact');
    tl.Position = [0.11 0.14 0.70 0.76];

    colormap(fig,parula(256));
    axesList = gobjects(4,1);

    for j = 1:4
        ax = nexttile(tl,j);
        axesList(j) = ax;

        p = R.selected{j};
        v = p.visual.(sliceKind);

        imagesc(ax,v.axis1,v.axis2,v.intensityNormalized.');
        axis(ax,'xy');
        axis(ax,'image');
        caxis(ax,[0 1]);

        set(ax,'FontName',fontName,'FontSize',fsTick, ...
            'TickDir','out','LineWidth',0.72,'Layer','top', ...
            'TickLength',[0.018 0.018]);

        set_compact_pi_ticks(ax,v.axis1,v.axis2);

        title(ax,sprintf('$S_%d$, mode %d',j,p.modeIndex), ...
            'Interpreter','latex', ...
            'FontName',fontName,'FontSize',fsTitle, ...
            'FontWeight','normal');

        % Keep tick labels only on the outside boundary of the 2x2 group.
        row = ceil(j/2);
        col = mod(j-1,2)+1;

        if row == 1
            ax.XTickLabel = [];
        end
        if col == 2
            ax.YTickLabel = [];
        end
    end

    % Outer shared labels only, so the tiles themselves can be larger.
    annotation(fig,'textbox',[0.38 0.02 0.14 0.05], ...
        'String',xLabel, ...
        'Interpreter','latex', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'LineStyle','none', ...
        'FontName',fontName,'FontSize',fsAxis);

    annotation(fig,'textbox',[0.01 0.40 0.05 0.14], ...
        'String',yLabel, ...
        'Interpreter','latex', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'LineStyle','none', ...
        'Rotation',90, ...
        'FontName',fontName,'FontSize',fsAxis);

    % Use a manually positioned slim colorbar instead of the tiledlayout
    % east tile, which wastes too much width.
    cb = colorbar(axesList(4),'eastoutside');
    cb.Ticks = [0 0.5 1];
    cb.FontName = fontName;
    cb.FontSize = fsColorbar;
    cb.TickDirection = 'out';
    cb.Units = 'normalized';
    cb.Position = [0.86 0.19 0.018 0.64];

    if saveFigure
        mediumName = medium_letter(R);
        base = sprintf('Figure_Exp4_Medium%s_Field_%s_S1S4_N5',mediumName,fileTag);
        exportgraphics(fig,fullfile(root,'figure',[base '.png']),'Resolution',600);
        exportgraphics(fig,fullfile(root,'figure',[base '.pdf']),'ContentType','vector');
        savefig(fig,fullfile(root,'figure',[base '.fig']));
    end
end

function set_compact_pi_ticks(ax,axis1,axis2)
    % Visualization planes are sampled over [-2*pi,2*pi]^2.
    tol = 5e-8;

    if abs(min(axis1)+2*pi)<tol && abs(max(axis1)-2*pi)<tol
        ax.XTick = [-2*pi 0 2*pi];
        ax.XTickLabel = {'$-2\pi$','$0$','$2\pi$'};
        ax.TickLabelInterpreter = 'latex';
    end

    if abs(min(axis2)+2*pi)<tol && abs(max(axis2)-2*pi)<tol
        ax.YTick = [-2*pi 0 2*pi];
        ax.YTickLabel = {'$-2\pi$','$0$','$2\pi$'};
        ax.TickLabelInterpreter = 'latex';
    end
end

function name = medium_letter(R)
    tag = char(R.cfg.output.tag);
    if strcmpi(tag,'MediumA') || strcmpi(tag,'A')
        name = 'A';
    elseif strcmpi(tag,'MediumB') || strcmpi(tag,'B')
        name = 'B';
    elseif strcmpi(tag,'PAPER')
        name = 'PAPER';
    else
        error('Unexpected medium tag: %s',tag);
    end
end