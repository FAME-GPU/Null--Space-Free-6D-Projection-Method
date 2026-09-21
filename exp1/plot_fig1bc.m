function plot_fig1bc()
%PLOT_FIG1BC Plot and save publication-style Fig. 1(b,c).
%
% Fig. 1(b): average linear iterations per outer Krylov step versus
%             original GEVP dimension 3N_F (log-log axes).
%             For each completed computation, the plotted value is
%             totalLinearSolverIterations / outerSteps.
% Fig. 1(c): solver wall time vs original GEVP dimension 3N_F
%             (log x-axis, linear y-axis).
% Fourier truncations N = 3:7 are shown.
%
% Result files are searched in terminal_results/ first and then results/.
% Expected files: fig1bc_A.mat, fig1bc_B.mat, fig1bc_C.mat.

    cfg    = make_config_exp1bc();
    root   = exp1bc_project_root();
    dataDir = resolve_fig1bc_result_dir();
    figDir = fullfile(root,cfg.output.figureDirectory);
    if ~exist(figDir,'dir'), mkdir(figDir); end

    methods = {'A','B','C'};
    styleA = make_style([0.0000,0.4470,0.7410],'o','-');
    styleB = make_style([0.8500,0.3250,0.0980],'s','-');
    styleC = make_style([0.9290,0.6940,0.1250],'^','-');
    styles = {styleA,styleB,styleC};

    data = struct();
    for j = 1:3
        file = fullfile(dataDir,sprintf('fig1bc_%s.mat',methods{j}));
        if ~exist(file,'file')
            error('Missing %s. Run Method %s first.',file,methods{j});
        end
        S = load(file,'results');
        data.(methods{j}) = normalize_one_method(S.results.scaleResults);
    end

    % Only show N = 3:7.
    showN = 3:7;

    %% Fig. 1(b): average linear iterations per outer step (log-log axes)
    fB = figure('Color','w','Units','inches','Position',[1.0 1.0 2.45 2.18], ...
                'PaperPositionMode','auto','Name','Fig1b_TotalLinearIterations');
    axB = axes('Parent',fB); hold(axB,'on'); box(axB,'on'); grid(axB,'on');
    format_axes(axB);
    axB.XScale = 'log';
    axB.YScale = 'log';
    axB.XLim = [1e5 3e7];
    axB.XTick  = [1e5 1e6 1e7];
    axB.YLim   = [1e1 3e3];
    axB.YTick  = [1e1 1e2 1e3];

    plot_success_curve(axB, data.A, showN, styles{1}, 'OSI', 'iterations');
    plot_success_curve(axB, data.B, showN, styles{2}, 'NRI', 'iterations');
    plot_success_curve(axB, data.C, showN, styles{3}, 'ERI', 'iterations');

    % Mark the first unresolved NRI scale by a dashed continuation to an
    % x-marker near the upper boundary.  The marker is deliberately not
    % interpreted as a numerical value; it indicates that the computation
    % did not complete at that problem size.
    annotate_failed_scale(axB, data.B, showN, styles{2}, 'iterations');

    xlabel(axB,'Original GEVP dimension $3N_F$', ...
        'Interpreter','latex','FontName','Times New Roman','FontSize',10.8);
    ylabel(axB,{'Average linear iterations',' per outer step'}, ...
        'FontName','Times New Roman','FontSize',10.8);
    lgB = legend(axB,{'OSI','NRI','ERI'}, ...
        'Location','northeast','Box','on','Interpreter','none');
    set(lgB,'FontName','Times New Roman','FontSize',8.8, ...
        'Color','white','EdgeColor',[0.20 0.20 0.20],'LineWidth',0.75);
    move_legend_down(lgB,0.05);

    save_figure_set(fB,fullfile(figDir,cfg.output.figureBBase),cfg.output.pngResolution);

    %% Fig. 1(c): solver wall time (log x-axis, linear y-axis)
    fC = figure('Color','w','Units','inches','Position',[1.0 1.0 2.45 2.18], ...
                'PaperPositionMode','auto','Name','Fig1c_WallTime');
    axC = axes('Parent',fC); hold(axC,'on'); box(axC,'on'); grid(axC,'on');
    format_axes(axC);
    axC.XScale = 'log';
    axC.XLim = [1e5 3e7];
    axC.XTick = [1e5 1e6 1e7];
    axC.YLim  = [0 5000];
    axC.YTick = 0:1000:5000;

    plot_success_curve(axC, data.A, showN, styles{1}, 'OSI', 'time');
    plot_success_curve(axC, data.B, showN, styles{2}, 'NRI', 'time');
    plot_success_curve(axC, data.C, showN, styles{3}, 'ERI', 'time');

    % Use the same censored-point convention as in panel (b): the first
    % unresolved NRI scale is connected to the last completed point by a
    % dashed segment and marked by an x near the upper boundary.
    annotate_failed_scale(axC, data.B, showN, styles{2}, 'time');

    xlabel(axC,'Original GEVP dimension $3N_F$', ...
        'Interpreter','latex','FontName','Times New Roman','FontSize',10.8);
    ylabel(axC,'Solver wall time (s)', ...
        'FontName','Times New Roman','FontSize',10.8);
    lgC = legend(axC,{'OSI','NRI','ERI'}, ...
        'Location','northwest','Box','on','Interpreter','none');
    set(lgC,'FontName','Times New Roman','FontSize',8.8, ...
        'Color','white','EdgeColor',[0.20 0.20 0.20],'LineWidth',0.75);
    % move_legend_down(lgC,0.03);

    save_figure_set(fC,fullfile(figDir,cfg.output.figureCBase),cfg.output.pngResolution);

    fprintf('Saved Fig. 1(b,c) to:\n  %s\n',figDir);
end

function S = make_style(color,marker,lineStyle)
    S.color = color;
    S.marker = marker;
    S.lineStyle = lineStyle;
    S.lineWidth = 1.55;
    S.markerSize = 5.4;
end

function format_axes(ax)
    set(ax,'FontName','Times New Roman', ...
           'FontSize',9.6, ...
           'LineWidth',0.95, ...
           'TickDir','out', ...
           'TickLength',[0.020 0.020], ...
           'Layer','top', ...
           'XMinorTick','off', ...
           'YMinorTick','off');
    ax.GridAlpha = 0.20;
    ax.MinorGridAlpha = 0.10;
end

function R = normalize_one_method(scaleResults)
%NORMALIZE_ONE_METHOD Extract quantities needed by the final Fig. 1(b,c).
%
% For completed runs, panel (b) uses
%
%   average linear iterations per outer step
%       = totalLinearSolverIterations / outerSteps.
%
% This uses the already stored terminal results; no eigensolve is rerun.

    if iscell(scaleResults)
        cells = scaleResults;
    else
        cells = arrayfun(@(s)s,scaleResults,'UniformOutput',false);
    end

    n = numel(cells);
    R = repmat(struct( ...
        'N',NaN, ...
        'status','', ...
        'iter',NaN, ...
        'time',NaN, ...
        'totalLinearIterations',NaN, ...
        'outerSteps',NaN),n,1);

    for i = 1:n
        s = cells{i};

        R(i).N = double(s.N);
        R(i).status = char(s.status);
        R(i).time = double(s.plotWallSeconds);

        % Completed runs in the current result files contain both
        % totalLinearSolverIterations and outerSteps.  Use those quantities
        % directly so OSI, NRI, and ERI are normalized by the same outer
        % Krylov-work measure.
        if isfield(s,'totalLinearSolverIterations')
            R(i).totalLinearIterations = double(s.totalLinearSolverIterations);
        elseif isfield(s,'plotLinearIterations')
            R(i).totalLinearIterations = double(s.plotLinearIterations);
        end

        if isfield(s,'outerSteps')
            R(i).outerSteps = double(s.outerSteps);
        elseif isfield(s,'inverseCalls')
            R(i).outerSteps = double(s.inverseCalls);
        end

        if strcmp(R(i).status,'success') && ...
                isfinite(R(i).totalLinearIterations) && ...
                isfinite(R(i).outerSteps) && R(i).outerSteps > 0
            R(i).iter = R(i).totalLinearIterations / R(i).outerSteps;
        end
    end
end

function plot_success_curve(ax,R,showN,sty,displayName,fieldKind)
    maskN = ismember([R.N],showN);
    Rv = R(maskN);
    ok = arrayfun(@(s) strcmp(s.status,'success'), Rv);
    Ns = [Rv(ok).N];
    Xs = original_dimension(Ns);
    if strcmp(fieldKind,'iterations')
        % R.iter is the average total linear-solver iterations per
        % completed outer Krylov step.
        Ys = [Rv(ok).iter];
    else
        Ys = [Rv(ok).time];
    end
    if isempty(Ns)
        return;
    end
    plot(ax,Xs,Ys, 'LineStyle',sty.lineStyle, 'Color',sty.color, ...
        'LineWidth',sty.lineWidth, 'Marker',sty.marker, ...
        'MarkerSize',sty.markerSize, 'MarkerEdgeColor',sty.color, ...
        'MarkerFaceColor','none', ...
        'DisplayName',displayName);
end

function annotate_failed_scale(ax,R,showN,sty,fieldKind)
%ANNOTATE_FAILED_SCALE Mark the first incomplete scale after completed runs.
%
% The last completed NRI point is connected to the first incomplete scale
% by a dashed segment.  An x-marker near the upper y-limit indicates that
% no completed value is available there; its vertical position is only a
% graphical censoring marker, not a measured quantity.

    maskN = ismember([R.N],showN);
    Rv = R(maskN);
    if isempty(Rv), return; end

    failedIdx = find(~strcmp({Rv.status},'success'),1,'first');
    if isempty(failedIdx) || failedIdx == 1
        return;
    end

    prevIdx = find(strcmp({Rv(1:failedIdx-1).status},'success'),1,'last');
    if isempty(prevIdx)
        return;
    end

    sPrev = Rv(prevIdx);
    sFail = Rv(failedIdx);

    xPrev = original_dimension(sPrev.N);
    xFail = original_dimension(sFail.N);

    if strcmp(fieldKind,'iterations')
        yPrev = sPrev.iter;
        % For the logarithmic averaged-iteration panel, keep the x-marker
        % slightly below the upper frame.  Its vertical position is only a
        % censoring symbol and is not a computed average.
        yFail = ax.YLim(2) / 1.10;
    else
        yPrev = sPrev.time;
        yFail = 0.96 * ax.YLim(2);
    end

    if ~isfinite(yPrev) || yPrev <= 0
        return;
    end

    % Dashed continuation: incomplete/censored scale, not a data curve.
    plot(ax,[xPrev xFail],[yPrev yFail], ...
        'LineStyle','--', ...
        'Color',sty.color, ...
        'LineWidth',1.35, ...
        'HandleVisibility','off');

    % Failure marker.  No text label is added: the dashed segment plus the
    % x-marker carries the same convention in panels (b) and (c).
    plot(ax,xFail,yFail,'x', ...
        'Color',sty.color, ...
        'LineWidth',1.55, ...
        'MarkerSize',7.8, ...
        'HandleVisibility','off');
end

function move_legend_down(lg,delta)
%MOVE_LEGEND_DOWN Shift a legend downward in normalized figure coordinates.
    drawnow;
    oldUnits = lg.Units;
    lg.Units = 'normalized';
    p = lg.Position;
    p(2) = max(0.02,p(2)-delta);
    lg.Position = p;
    lg.Units = oldUnits;
end

function x = original_dimension(N)
%ORIGINAL_DIMENSION Dimension of the original Maxwell GEVP: 3 N_F.
    x = 3 .* (2 .* double(N)).^6;
end

function lim = original_dimension_bounds(showN)
%ORIGINAL_DIMENSION_BOUNDS Add a small logarithmic margin around N=3:7.
    xmin = original_dimension(min(showN));
    xmax = original_dimension(max(showN));
    pad = 10^0.05;
    lim = [xmin/pad, xmax*pad];
end

function save_figure_set(fig,base,resolution)
    drawnow;
    savefig(fig,[base '.fig']);
    if exist('exportgraphics','file') == 2
        exportgraphics(fig,[base '.png'],'Resolution',resolution,'BackgroundColor','white');
        exportgraphics(fig,[base '.pdf'],'ContentType','vector','BackgroundColor','white');
    else
        print(fig,[base '.png'],'-dpng',sprintf('-r%d',resolution));
        print(fig,[base '.pdf'],'-dpdf');
    end
    if ~exist([base '.fig'],'file') || ~exist([base '.png'],'file')
        error('Figure export failed for %s.',base);
    end
end
