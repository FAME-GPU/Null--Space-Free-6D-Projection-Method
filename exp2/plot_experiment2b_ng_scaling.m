function plot_experiment2b_ng_scaling(results,cfg,rootDir)
%PLOT_EXPERIMENT2B_NG_SCALING Final manuscript Figure 2(b).
%
% Horizontal axis:
%   Cube of Yee half-width n^3, with N_G=(2n+3)^3 points per field component.
%
% Left axis:
%   GPU reconstruction time for blockwise direct and radius-controlled
%   p=10 multi-center Taylor reconstruction.
%
% Right axis:
%   Taylor relative field error e2 wherever a complete direct reference
%   exists.
%
% Timeout policy:
%   A timeout_incomplete point is NOT treated as a completed timing datum.
%   The completed timing curve stops at the last completed size, while the
%   timeout location is shown separately at the prescribed time budget.

    if nargin < 3 || isempty(rootDir)
        rootDir = fileparts(mfilename('fullpath'));
    end

    figDir = fullfile(rootDir,'figures');
    if ~exist(figDir,'dir')
        mkdir(figDir);
    end

    T = results.scaling;

    fontLabel  = 16;
    fontTick   = 14;
    fontLegend = 13;
    lineW      = 2.0;
    markerSize = 7.5;

    baseColors = get(groot,'defaultAxesColorOrder');
    if size(baseColors,1) < 3
        baseColors = lines(3);
    end
    cD = baseColors(1,:);
    cT = baseColors(2,:);
    cE = baseColors(3,:);

    % Use painters from figure creation onward.  This is preferable for the
    % present 2-D manuscript graphic and also avoids unnecessary OpenGL use
    % on headless terminal nodes.
    f = figure('Color','w', ...
        'Units','inches', ...
        'Position',[1 1 5.60 4.35], ...
        'Renderer','painters');

    set(f,'PaperUnits','inches', ...
        'PaperPosition',[0 0 5.60 4.35], ...
        'PaperSize',[5.60 4.35], ...
        'PaperPositionMode','manual');

    mainAxPos = [0.12 0.29 0.73 0.63];
    ax = axes(f,'Position',mainAxPos);
    hold(ax,'on');
    box(ax,'on');
    grid(ax,'on');

    T = T(T.n>=40,:); % manuscript range; reference includes an extra n=20 row
    x = double(T.n(:)).^3;
    budget = cfg.scaling.timeoutSeconds;

    %% =========================================================
    %  Left axis: reconstruction time
    %  =========================================================
    yyaxis(ax,'left');
    ax.YAxis(1).Scale = 'log';

    % Plot only COMPLETED timings as timing data.  This prevents an
    % incomplete timeout point from being connected as though it were a
    % measured reconstruction time.
    dCompleted = logical(T.DirectCompleted) & ...
        isfinite(T.DirectTimeSeconds) & T.DirectTimeSeconds > 0;
    tCompleted = logical(T.TaylorCompleted) & ...
        isfinite(T.TaylorTimeSeconds) & T.TaylorTimeSeconds > 0;

    hD = plot(ax,x(dCompleted),T.DirectTimeSeconds(dCompleted),'-o', ...
        'Color',cD, ...
        'LineWidth',lineW, ...
        'MarkerSize',markerSize, ...
        'MarkerFaceColor',cD, ...
        'MarkerEdgeColor',cD, ...
        'DisplayName','Blockwise direct');

    hT = plot(ax,x(tCompleted),T.TaylorTimeSeconds(tCompleted),'-s', ...
        'Color',cT, ...
        'LineWidth',lineW, ...
        'MarkerSize',markerSize, ...
        'MarkerFaceColor',cT, ...
        'MarkerEdgeColor',cT, ...
        'DisplayName','Multi-center Taylor');

    % Time-budget reference line.
    if ~isempty(x)
        plot(ax,[min(x) max(x)],[budget budget],':', ...
            'Color',[0.45 0.45 0.45], ...
            'LineWidth',1.0, ...
            'HandleVisibility','off');
    end

    % Incomplete timeout points are shown separately at the budget level.
    % Subsequent skipped sizes are intentionally not shown for that method.
    dTimeout = (string(T.DirectStatus) == "timeout_incomplete");
    tTimeout = (string(T.TaylorStatus) == "timeout_incomplete");

    if any(dTimeout)
        xd = x(dTimeout);
        plot(ax,xd,budget*ones(size(xd)),'x', ...
            'Color',cD, ...
            'MarkerSize',markerSize+1.5, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.6, ...
            'HandleVisibility','off');

        % Usually only one timeout point exists because all larger sizes are
        % skipped.  Label every timeout robustly in case the policy changes.
        for j = 1:numel(xd)
            text(ax,xd(j),budget*1.08,'$>10^4\,$s', ...
                'Interpreter','latex', ...
                'Color',cD, ...
                'FontSize',fontLegend-1, ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','bottom');
        end
    end

    if any(tTimeout)
        xt = x(tTimeout);
        plot(ax,xt,budget*ones(size(xt)),'x', ...
            'Color',cT, ...
            'MarkerSize',markerSize+1.5, ...
            'MarkerFaceColor','w', ...
            'LineWidth',1.6, ...
            'HandleVisibility','off');

        for j = 1:numel(xt)
            text(ax,xt(j),budget*1.08,'$>10^4\,$s', ...
                'Interpreter','latex', ...
                'Color',cT, ...
                'FontSize',fontLegend-1, ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','bottom');
        end
    end

    ylabel(ax,'GPU reconstruction time (s)', ...
        'Interpreter','latex', ...
        'FontSize',fontLabel);

    ax.YMinorGrid = 'on';

    % The experiment uses 10^4 s as the timing budget.  There is no need to
    % leave an unused full decade up to 10^5 s.  A factor-two headroom keeps
    % timeout markers/labels visible while using the plotting area well.
    validTimes = [T.DirectTimeSeconds(dCompleted); T.TaylorTimeSeconds(tCompleted)];
    validTimes = validTimes(isfinite(validTimes) & validTimes > 0);
    if isempty(validTimes)
        yLo = 1;
    else
        yLo = 10^floor(log10(min(validTimes)));
    end
    yHi = max(2*budget, 1.25*max([validTimes; budget]));
    ylim(ax,[yLo yHi]);

    %% =========================================================
    %  Right axis: Taylor relative field error
    %  =========================================================
    yyaxis(ax,'right');
    ax.YAxis(2).Scale = 'log';

    eMask = isfinite(T.RelativeFieldError2) & T.RelativeFieldError2 > 0;
    hE = plot(ax,x(eMask),T.RelativeFieldError2(eMask),'--d', ...
        'Color',cE, ...
        'LineWidth',1.7, ...
        'MarkerSize',markerSize, ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',cE, ...
        'DisplayName','Taylor error $e_2$');

    ylabel(ax,'Taylor relative field error $e_2$', ...
        'Interpreter','latex', ...
        'FontSize',fontLabel);

    e = T.RelativeFieldError2(eMask);
    if ~isempty(e)
        elo = 10^floor(log10(min(e)));
        ehi = 10^ceil(log10(max(e)));
        if elo == ehi
            elo = elo/10;
            ehi = ehi*10;
        end
        ylim(ax,[elo ehi]);
    else
        ylim(ax,[1e-12 1e-8]);
    end
    ax.YMinorGrid = 'on';

    %% =========================================================
    %  Shared x axis / legend
    %  =========================================================
    xlabel(ax,'Yee half-width cubed $n^3$', ...
        'Interpreter','latex', ...
        'FontSize',fontLabel);

    set(ax, ...
        'FontSize',fontTick, ...
        'LineWidth',1.05, ...
        'TickDir','out', ...
        'XColor','k');

    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = 'k';

    % n is sampled uniformly: 20,40,...,200.  Keep every tested value as a
    % tick so the sampling sequence is visible directly from the figure.
    ax.XScale='log';
    xticks(ax,[40 80 120 160 200].^3);
    xlim(ax,[min(x)/1.15 max(x)*1.15]);

    lgd = legend(ax,[hD hT hE], ...
        {'Blockwise direct','Multi-center Taylor','Taylor error $e_2$'}, ...
        'Interpreter','latex', ...
        'FontSize',fontLegend, ...
        'Location','southeast');
    lgd.Box = 'on';

    drawnow;
    export_all(f,fullfile(figDir,cfg.output.figureBase),cfg.output.pngResolution);
    close(f);
end

function export_all(fig,base,res)
    drawnow;
    set(fig,'Renderer','painters');

    print(fig,[base '.eps'],'-depsc2','-painters');

    if exist('exportgraphics','file') == 2
        exportgraphics(fig,[base '.png'], ...
            'Resolution',res, ...
            'BackgroundColor','white');
    else
        print(fig,[base '.png'],'-dpng',sprintf('-r%d',res));
    end

    savefig(fig,[base '.fig']);
end
