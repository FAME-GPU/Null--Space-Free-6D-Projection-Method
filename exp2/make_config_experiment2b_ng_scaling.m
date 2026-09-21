function cfg = make_config_experiment2b_ng_scaling()
%MAKE_CONFIG_EXPERIMENT2B_NG_SCALING Configuration for Figure 2(b) scaling.
%
% The experiment fixes the N=5 Fourier discretization and one Fourier
% eigenvector (mode 10), then increases the Yee reconstruction window while
% keeping h fixed.  The multi-center Taylor method uses p=10 and, for each
% window size, the smallest number of cubic local blocks whose exact phase
% radius does not exceed the adopted n=40, 4^3-center radius.

    %% GPU
    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.precision = 'double';

    %% Physical / lifted problem -- identical to current Experiment 2
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

    %% Smooth scalar heterogeneous permittivity
    cfg.material.epsC = 4;
    cfg.material.alpha = 1.2;

    %% Exact scalar rank-1 Fourier mass backend
    cfg.mass.backend = 'rank1-exact-scalar';
    cfg.mass.rank1.N5.M = 24760990;
    cfg.mass.rank1.N5.z = [1 19 361 6859 130321 2476099];

    %% Spectrum / explicit inverse Lanczos
    cfg.spectrum.numEig = 10;
    cfg.spectrum.modes = 1:10;
    cfg.spectrum.performanceMode = 10;

    cfg.solver.tolCG = 1e-10;
    cfg.solver.tolLanczos = 1e-10;
    cfg.solver.krylovDim = 40;
    cfg.solver.maxLanczosRestarts = 30;
    cfg.solver.maxitInnerCG = 3000;
    cfg.solver.tolRecoverM = 1e-12;
    cfg.solver.maxitRecoverM = 3000;
    cfg.solver.rngSeed = 20260906;
    cfg.solver.verboseLanczos = false;
    cfg.solver.requiredOriginalGEVPResidual = 1e-10;

    %% Figure 2(b) Yee-size sweep
    cfg.scaling.nValues = [40 60 80 100 120 140 160 180 200];
    cfg.scaling.h = 0.0125;

    % Taylor accuracy regime.  IMPORTANT: the radius threshold is not the
    % rounded number 5.18.  At runtime it is computed exactly from the same
    % qModes and from the adopted n=40, k=4 configuration used in panel (a).
    cfg.scaling.taylorOrder = 10;
    cfg.scaling.baselineN = 40;
    cfg.scaling.baselineBlocksPerDimension = 4;  % 4^3 = 64 centers
    cfg.scaling.radiusRelativeTolerance = 5e-13;
    cfg.scaling.termBatchSizeRequested = 32;

    % Each method has its own independent time budget.  If a method reaches
    % this budget before finishing the current size, it is stopped at the
    % next safe block-group boundary and is not attempted for larger n.
    cfg.scaling.timeoutSeconds = 1.0e4;
    cfg.scaling.centersPerTimeoutCheck = 32;

    % Direct dense-phase evaluation is always blocked.  The largest safe
    % batch is chosen from this list using live GPU AvailableMemory.
    cfg.scaling.directBlockSizeRequested = 256;
    cfg.scaling.directBlockCandidates = [256 192 128 96 64 48 32 24 16 8];

    % A small warm-up avoids charging one-time GPU kernel initialization to
    % the first scaling point.  Warm-up uses only one local block.
    cfg.scaling.warmupKernels = true;

    %% Memory-safety policy
    % No full N_G-by-3 Yee coordinate arrays and no full reconstructed fields
    % are ever stored.  Coordinates and field values are streamed by local
    % spatial block.  The reserve below protects MATLAB/GPU runtime overhead.
    cfg.memory.reserveGiB = 2.0;
    cfg.memory.directWorkspaceSafetyFactor = 1.20;
    cfg.memory.taylorWorkspaceSafetyFactor = 1.25;
    cfg.memory.minAvailableGiBForRun = 3.0;

    %% Output / run switches
    cfg.run.solve = true;
    cfg.run.makePlot = true;
    cfg.run.resume = true;
    cfg.output.verbose = true;
    cfg.output.pngResolution = 600;

    cfg.output.resultFile = 'Experiment2b_N5_NgScaling_results.mat';
    cfg.output.checkpointFile = 'Experiment2b_N5_NgScaling_checkpoint.mat';
    cfg.output.scalingCSV = 'Experiment2b_N5_NgScaling.csv';
    cfg.output.summaryTXT = 'Experiment2b_N5_NgScaling_summary.txt';
    cfg.output.figureBase = 'Figure_Exp2b_NgScaling_N5_GPU';
end
