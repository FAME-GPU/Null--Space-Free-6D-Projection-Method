function cfg = make_config_experiment4()
%MAKE_CONFIG_EXPERIMENT4 Final all-GPU N=5 physical-space closure/LWRQ study.
%
% Large numerical kernels (eigensolve, reconstruction, cropped Yee action,
% material sampling and LWRQ convolution/reductions) are executed on GPU.
% CPU is used only for small metadata, checkpoint serialization and plotting.

    %% GPU
    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.precision = 'double';
    cfg.gpu.largeWindowSafetyGiB = 2.0;

    %% 6D problem
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
    cfg.problem.q = [0.13;0.21;0.17];

    %% Material
    cfg.material.epsC = 4;
    cfg.material.alpha = 1.2;

    %% Exact rank-1 scalar mass backend
    cfg.mass.backend = 'rank1-exact-scalar';
    cfg.mass.rank1.N5.M = 24760990;
    cfg.mass.rank1.N5.z = [1 19 361 6859 130321 2476099];

    %% Tracked modes
    cfg.tracking.n2Modes = [1 10];
    cfg.tracking.n5Modes = [11 114];
    cfg.tracking.n2Lambda = [1.201276300237e-02 4.496454358674e-02];
    cfg.tracking.n5LambdaReference = [1.200090550277e-02 4.462939906027e-02];
    cfg.tracking.fourierOverlap = [0.99078125 0.94844066];
    cfg.tracking.commonEnergyCapture = [0.99138739 0.99402804];
    cfg.tracking.pairPrincipalCosines = [0.99078126 0.94844074];
    cfg.tracking.lambdaCheckRelTol = 5e-7;

    %% Selected-mode eigensolve
    cfg.scan.N5.k = 140;
    cfg.scan.N5.pExtra = 48;
    cfg.scan.N5.candidateHalfWidth = 0;

    cfg.solver.tolCG = 1e-10;
    cfg.solver.tolLanczos = 1e-10;
    cfg.solver.maxLanczosRestarts = 30;
    cfg.solver.maxitInnerCG = 3000;
    cfg.solver.tolRecoverM = 1e-12;
    cfg.solver.maxitRecoverM = 3000;
    cfg.solver.rngSeed = 20260831;
    cfg.solver.verboseLanczos = false;
    cfg.solver.requiredSelectedRelRitzResidual = 1e-10;

    %% Taylor reconstruction
    cfg.taylor.order = 10;
    cfg.taylor.termBatchSize = 32;
    cfg.taylor.centerBatchSize = 8;             % standard mesh cases
    cfg.taylor.largeWindowCenterBatchSize = 32; % streaming L=4 reconstruction
    cfg.taylor.phaseRadiusTarget = 5.20;
    cfg.taylor.maxBlocksPerDimension = 32;
    cfg.taylor.phaseRadiusSlack = 2e-12;

    %% LWRQ
    cfg.lwrq.sigmaFactor = 2;
    cfg.lwrq.neighborRadiusCells = 3;
    cfg.lwrq.massFloorRelative = 1e-14;

    %% Mesh refinement: fixed physical half-width 0.5
    cfg.mesh.h = [0.1 0.05 0.025 0.0125];
    cfg.mesh.n = [5 10 20 40];

    %% Window study: one h=0.025 master reconstruction to half-width 4.
    % The smaller windows are exact centered crops of that same master field.
    cfg.window.h = 0.025;
    cfg.window.n = [10 15 20 30 40 60 80 100 120 140 160];
    cfg.window.masterN = max(cfg.window.n);

    %% Default reference setting retained for tables / manuscript text.
    cfg.default.h = 0.025;
    cfg.default.n = 20;

    %% Run/checkpoint policy
    cfg.run.reuseSelectedEigenpairCheckpoint = true;
    cfg.run.saveSelectedEigenpairCheckpoint = true;
    cfg.run.saveMaxWindowSpatialSlices = true;
    cfg.run.validateLargeWindowStreamingAtDefault = true;
    cfg.run.streamingValidationRelTol = 5e-11;
    cfg.run.requireGPUPhysicalKernels = true;

    %% Outputs
    cfg.output.verbose = true;
    cfg.output.resultFile = 'Experiment3_N5_AllGPU_results.mat';
    cfg.output.checkpointFile = 'Experiment4_N5_selected_eigenpairs_checkpoint.mat'; % validated legacy filename
    cfg.output.defaultCSV = 'Experiment3_N5_default_modes.csv';
    cfg.output.meshCSV = 'Experiment3_N5_mesh_sensitivity.csv';
    cfg.output.windowCSV = 'Experiment3_N5_window_sensitivity.csv';
    cfg.output.detailedCSV = 'Experiment3_N5_all_mode_cases.csv';
    cfg.output.summaryTXT = 'Experiment3_N5_AllGPU_summary.txt';
    cfg.output.latexTXT = 'Experiment3_N5_AllGPU_LaTeX_values.txt';
end
