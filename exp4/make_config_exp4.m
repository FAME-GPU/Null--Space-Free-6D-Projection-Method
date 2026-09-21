function cfg = make_config_exp4(mediumName)
%MAKE_CONFIG_EXP4 GPU production configuration for Experiment 4.
%
% Medium A (unchanged):
%   E_A(x) = 4 exp[f_A(x)] I_3,
%   f_A = (1.2/6) sum_l cos(x_l)
%       + (0.3/3)[cos(x1+x4)+cos(x2+x5)+cos(x3+x6)].
%
% Medium B (low-frequency beat-product model):
%   E_B(x) = 1.5 [2.8
%       + cos(x1-x4) cos(x2-x5)
%       + 0.8 cos(x2-x5) cos(x3-x6)
%       + 0.6 cos(x3-x6) cos(x1-x4)] I_3.
% A and B are independent production jobs. All large spectral, material,
% reconstruction, Yee, and LWRQ calculations remain on the GPU.

    if nargin<1 || isempty(mediumName), mediumName='PAPER'; end
    mediumName=upper(char(mediumName));
    if ~ismember(mediumName,{'A','B','PAPER'})
        error('mediumName must be ''A'', ''B'', or ''PAPER''.');
    end

    %% GPU -- strict production policy
    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.precision = 'double';
    cfg.gpu.strictLargeArrayGPU = true;
    cfg.gpu.gatherOnlyForIO = true;

    %% 6D embedding
    cfg.problem.dim = 6;
    cfg.problem.N = 5;
    cfg.problem.T = 2*pi*ones(6,1);
    cfg.problem.P = [ ...
        1       0       0; ...
        0       1       0; ...
        0       0       1; ...
        sqrt(2) 0       0; ...
        0       sqrt(3) 0; ...
        0       0       sqrt(5)];

    %% Two quasiperiodic scalar media
    cfg.material.name = mediumName;
    cfg.material.referenceEpsC = 4;
    cfg.material.meanQuadratureN = 256;

    if strcmp(mediumName,'A')
        cfg.material.modelType = 'reference_A';
        cfg.material.alpha = 1.2;
        cfg.material.beta = 0.3;
        cfg.material.label = 'Medium A: reference exponential model';
    else
    cfg.material.modelType = 'mixed_fourier_B';

    % Medium B: low-frequency beat-product model
    %
    % E_B(x) = cB * [ b0
    %   + a12*cos(x1-x4)*cos(x2-x5)
    %   + a34*cos(x2-x5)*cos(x3-x6)
    %   + a56*cos(x3-x6)*cos(x1-x4) ] I_3.
    %
    % The parameter names a12/a34/a56 are retained to keep the
    % checkpoint/signature infrastructure unchanged.
    cfg.material.cB = 1.5;
    cfg.material.b0 = 2.8;
    cfg.material.a12 = 1.0;
    cfg.material.a34 = 0.8;
    cfg.material.a56 = 0.6;

    % Retained only for checkpoint-signature compatibility; unused by B2.
    cfg.material.a135 = 0.0;
    cfg.material.axialAmplitude = 0;
    if strcmp(mediumName,'PAPER')
        cfg.material.axialAmplitude = 0.1; % manuscript equation (6.2)
    end

    cfg.material.label = 'Medium B: legacy beat-product model';
    if strcmp(mediumName,'PAPER')
        cfg.material.label = 'Manuscript equation (6.2): beat products plus axial harmonics';
    end
end

    %% Exact scalar rank-1 mass backend for N=5
    cfg.mass.backend = 'rank1-exact-scalar';
    cfg.mass.rank1.N5.M = 24760990;
    cfg.mass.rank1.N5.z = [1 19 361 6859 130321 2476099];

    %% Closed physical Bloch path
    cfg.path.nodes = [ ...
        0.05 0.25 0.25 0.25 0.05; ...
        0.04 0.04 0.20 0.20 0.04; ...
        0.03 0.03 0.03 0.18 0.03];
    cfg.path.subintervalsPerSegment = 20;

    %% Low-frequency spectrum
    cfg.spectrum.nev = 20;
    cfg.spectrum.pExtra = 48;
    cfg.spectrum.p = cfg.spectrum.nev + cfg.spectrum.pExtra;
    cfg.spectrum.requireFullConvergence = true;

    %% Explicit inverse Lanczos / recovery
    cfg.solver.tolCG = 1e-10;
    cfg.solver.tolLanczos = 1e-10;
    cfg.solver.maxLanczosRestarts = 30;
    cfg.solver.maxitInnerCG = 3000;
    cfg.solver.tolRecoverM = 1e-12;
    cfg.solver.maxitRecoverM = 3000;
    cfg.solver.rngSeed = 20260902;
    cfg.solver.verboseLanczos = false;

    %% Four representative physical points, identical in A and B
    m = cfg.path.subintervalsPerSegment;
    if mod(m,2)~=0, error('subintervalsPerSegment must be even for midpoint selection.'); end
    cfg.selected.pointIndices = 1 + (0:3)*m + m/2;
    cfg.selected.modeIndices  = [1 9 12 18];
    cfg.selected.labels = {'S1','S2','S3','S4'};

    %% 3D LWRQ reconstruction
    cfg.taylor.order = 10;
    cfg.taylor.termBatchSize = 32;
    cfg.taylor.centerBatchSize = 8;
    cfg.taylor.phaseRadiusTarget = 5.20;
    cfg.taylor.maxBlocksPerDimension = 20;
    cfg.taylor.phaseRadiusSlack = 2e-12;

    cfg.lwrq.h = 0.025;
    cfg.lwrq.n = 20;
    cfg.lwrq.sigmaFactor = 2;
    cfg.lwrq.neighborRadiusCells = 3;
    cfg.lwrq.massFloorRelative = 1e-14;

    %% Enlarged 2D field slices for visualization
    cfg.visual.halfWidth = 2*pi;
    cfg.visual.numPoints = 161;
    cfg.visual.planes = {'z0','x0','xy'};
    cfg.visual.phaseRadiusTarget = 5.20;
    cfg.visual.maxBlocksPerDimension = 80;
    cfg.visual.centerBatchSize = 8;

    %% Checkpoint/resume
    cfg.run.reuseQCheckpoints = true;
    cfg.run.saveQCheckpoints = true;
    cfg.run.reusePostCheckpoints = true;
    cfg.run.savePostCheckpoints = true;

    %% Concise Experiment-4 output names
    cfg.output.verbose = true;
    cfg.output.tag = mediumName;
    cfg.output.resultFile = sprintf('Exp4_%s_results.mat',mediumName);
    cfg.output.summaryTXT = sprintf('Exp4_%s_summary.txt',mediumName);
    cfg.output.spectrumCSV = sprintf('Exp4_%s_spectrum.csv',mediumName);
    cfg.output.selectedCSV = sprintf('Exp4_%s_selected.csv',mediumName);
    cfg.output.timingCSV = sprintf('Exp4_%s_timing.csv',mediumName);
end
