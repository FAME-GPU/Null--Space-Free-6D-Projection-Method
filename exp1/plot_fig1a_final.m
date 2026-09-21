function plot_fig1a_final()
%PLOT_FIG1A_A_ONLY Final Experiment-1 Fig. 1(a): Methods A/B/C.
%
% Historical function name retained so START_PLOT_A does not need to change.
% This routine now plots all THREE methods in the final manuscript panel.
%
% Expected terminal results:
%   terminal_results/fig1a_A.mat
%   terminal_results/fig1a_B.mat
%   terminal_results/fig1a_C.mat
%
% Each MAT file must contain variable "results" with 12 independent Bloch
% points in results.qResults. The first point is repeated only for plotting
% so that the displayed path is closed.
%
% Plotted metric:
%   A: number of Top vector applications requested by MATLAB eigs;
%   B/C: number of outer K_r^{-1} applications in inverse Lanczos.
%
% Final panel styling deliberately matches the previous one-row/three-column
% Experiment-1 figure: compact 2.42 x 2.65 inch panel, manuscript-scale fonts,
% clear line/marker weights, endpoint Bloch labels, and horizontal legend.
%
% Output:
%   figures/Figure_Exp1a_OuterApplications_N5.fig
%   figures/Figure_Exp1a_OuterApplications_N5.png
%   figures/Figure_Exp1a_OuterApplications_N5.pdf

    root = locate_project_root();
    terminalDir = fullfile(root,'results');
    figDir = fullfile(root,'figures');

    if ~exist(terminalDir,'dir'), mkdir(terminalDir); end
    if ~exist(figDir,'dir'), mkdir(figDir); end

    fileA = fullfile(terminalDir,'fig1a_A.mat');
    fileB = fullfile(terminalDir,'fig1a_B.mat');
    fileC = fullfile(terminalDir,'fig1a_C.mat');

    assert_result_exists(fileA,'A');
    assert_result_exists(fileB,'B');
    assert_result_exists(fileC,'C');

    [pathA,outerA] = load_one_curve(fileA,'A');
    [pathB,outerB] = load_one_curve(fileB,'B');
    [pathC,outerC] = load_one_curve(fileC,'C');

    % Verify all three result files correspond to the same 12-point path.
    if max(abs(pathA(:)-pathB(:))) > 1e-12 || ...
       max(abs(pathA(:)-pathC(:))) > 1e-12
        error(['The Bloch-path coordinates in fig1a_A/B/C.mat are inconsistent. ', ...
               'Use results from the same 12-point Experiment-1 path.']);
    end

    % ---------------------------------------------------------------------
    % Closed display path. The eigensolve contains 12 independent points;
    % the first point is repeated only for drawing q^(0) at the right end.
    x = 1:13;
    yA = [outerA(:); outerA(1)];
    yB = [outerB(:); outerB(1)];
    yC = [outerC(:); outerC(1)];

    % Styling copied from the previous final Experiment-1 plotting routine.
    figW = 2.45;
    figH = 2.18;
    fontName = 'Times New Roman';
    fsTick   = 10.2;
    fsLabel  = 10.8;
    fsLegend = 9.0;
    lw       = 1.45;
    msA      = 4.8;
    msB      = 6.2;   % larger square keeps B visible when B and C coincide
    msC      = 4.8;

    endpointTicks  = [1 4 7 10 13];
    endpointLabels = {'$q^{(0)}$','$q^{(1)}$','$q^{(2)}$','$q^{(3)}$','$q^{(0)}$'};

    % Fixed method colors matching the previous figure.
    cA = [0.0000 0.4470 0.7410];
    cB = [0.8500 0.3250 0.0980];
    cC = [0.9290 0.6940 0.1250];

    fig = figure('Units','inches','Position',[0.4 0.4 figW figH], ...
        'Color','w','Name','Experiment 1 Fig. 1(a)');
    ax = axes(fig);
    hold(ax,'on');
    box(ax,'on');

    hA = plot(ax,x,yA,'-o', ...
        'Color',cA,'LineWidth',lw, ...
        'MarkerSize',msA,'MarkerFaceColor','none');

    % B and C may coincide pointwise. The larger hollow square together with
    % the smaller triangle lets both curves remain visible without shifting
    % either method away from its true data value.
    hB = plot(ax,x,yB,'--s', ...
        'Color',cB,'LineWidth',lw, ...
        'MarkerSize',msB,'MarkerFaceColor','none');

    hC = plot(ax,x,yC,'-^', ...
        'Color',cC,'LineWidth',lw, ...
        'MarkerSize',msC,'MarkerFaceColor','none');

    set(ax, ...
        'XLim',[1 13], ...
        'XTick',endpointTicks, ...
        'XTickLabel',endpointLabels, ...
        'TickLabelInterpreter','latex', ...
        'FontName',fontName, ...
        'FontSize',fsTick, ...
        'TickDir','out', ...
        'LineWidth',0.85, ...
        'Layer','top');

    % Keep exactly the same vertical scale as the previous manuscript panel.
    ylim(ax,[60 100]);
    yticks(ax,60:10:100);

    grid(ax,'on');
    ax.GridAlpha = 0.12;
    ax.MinorGridAlpha = 0.08;

    % "Outer operator applications" is the correct common metric now that
    % Method A uses MATLAB eigs rather than the previous custom Lanczos code.
    ylabel(ax,'Arnoldi/Lanczos iterations', ...
        'FontName',fontName,'FontSize',fsLabel);
    xlabel(ax,'Bloch path', ...
        'FontName',fontName,'FontSize',fsLabel);

    lg = legend(ax,[hA hB hC],{'OSI','NRI','ERI'}, ...
        'Location','north', ...
        'Orientation','horizontal', ...
        'NumColumns',3, ...
        'FontName',fontName, ...
        'FontSize',fsLegend, ...
        'Box','on');
    set(lg,'Color','white','EdgeColor',[0.20 0.20 0.20],'LineWidth',0.75);
    lg.ItemTokenSize = [11 8];

    % No MATLAB title and no '(a)' mark: LaTeX supplies the subcaption, exactly
    % as in the previous final three-panel Experiment-1 figure.

    baseName = 'Figure_Exp1a_OuterApplications_N5';
    figFile = fullfile(figDir,[baseName '.fig']);
    pngFile = fullfile(figDir,[baseName '.png']);
    pdfFile = fullfile(figDir,[baseName '.pdf']);

    savefig(fig,figFile);
    exportgraphics(fig,pngFile,'Resolution',600);
    exportgraphics(fig,pdfFile,'ContentType','vector');

    fprintf('Saved final Experiment-1 Fig. 1(a):\n');
    fprintf('  %s\n',figFile);
    fprintf('  %s\n',pngFile);
    fprintf('  %s\n',pdfFile);
end

function root = locate_project_root()
    root = fileparts(which('START_PLOT_FIG1A'));
    if isempty(root)
        root = fileparts(mfilename('fullpath'));
    end
    if isempty(root) || ~isfolder(root)
        root = pwd;
    end
end

function assert_result_exists(fileName,methodName)
    if ~exist(fileName,'file')
        error(['Missing formal terminal result for Method %s:\n  %s\n', ...
               'Copy fig1a_%s.mat into terminal_results/ and run START_PLOT_A again.'], ...
               methodName,fileName,methodName);
    end
end

function [pathCoordinate,outer] = load_one_curve(resultFile,methodName)
    S = load(resultFile,'results');
    if ~isfield(S,'results')
        error('File %s does not contain variable ''results''.',resultFile);
    end

    r = S.results;
    if ~isfield(r,'qResults') || numel(r.qResults) ~= 12
        error('Unexpected Fig. 1(a) result structure in %s.',resultFile);
    end

    qResults = r.qResults;
    if isstruct(qResults)
        qResults = arrayfun(@(s)s,qResults,'UniformOutput',false);
    end

    pathCoordinate = zeros(12,1);
    outer = zeros(12,1);

    for k = 1:12
        qk = qResults{k};

        if ~isfield(qk,'pathCoordinate')
            error('Method %s point %d is missing ''pathCoordinate''.',methodName,k);
        end
        pathCoordinate(k) = double(qk.pathCoordinate);
        outer(k) = extract_outer_count(qk,methodName,k);
    end
end

function val = extract_outer_count(qk,methodName,k)
    % Support the field names used by the A-only code and by previous B/C
    % Experiment-1 drivers without changing any numerical result file.
    candidateFields = { ...
        'outerSteps', ...
        'outerOperatorApplications', ...
        'outerApplications', ...
        'numOuterApplications', ...
        'outerIterations', ...
        'outerIters', ...
        'lanczosIterations', ...
        'outerIterationCount'};

    val = [];
    for j = 1:numel(candidateFields)
        f = candidateFields{j};
        if isfield(qk,f)
            val = double(qk.(f));
            break;
        end
    end

    if isempty(val)
        error(['Method %s point %d does not contain a recognized outer-count field. ', ...
               'Checked: %s'],methodName,k,strjoin(candidateFields,', '));
    end

    if ~isscalar(val) || ~isfinite(val) || val < 0
        error('Method %s point %d has an invalid outer-operation count.',methodName,k);
    end
end
