function cfg = make_config_experiment2_allgpu()
%MAKE_CONFIG_EXPERIMENT2_ALLGPU Final all-GPU reconstruction experiment.
%
% Numerical kernels for the eigensolver, blockwise direct reconstruction,
% multi-center Taylor reconstruction, cropped Yee action, and error norms
% remain on the GPU.  The CPU is used only for small metadata (block index
% lists), scalar diagnostics, saving, and plotting.

    %% GPU
    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.precision = 'double';

    %% Physical/lifted problem
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

    %% Yee reconstruction grid
    cfg.yee.n = 40;
    cfg.yee.h = 0.0125;

    %% GPU blockwise direct reference
    % 256 is retained for a conservative V100 memory footprint (~5.72 GiB
    % for the phase/exponential workspace per component, excluding inputs).
    cfg.direct.blockSize = 256;
    cfg.direct.warmupReference = false;

    %% Multi-center Taylor settings
    cfg.taylor.termBatchSize = 32;
    cfg.taylor.singleCenter = [0 0 0];
    cfg.taylor.performanceOrder = 10;
    cfg.taylor.blocksPerDimension = 4;   % 4^3 = 64 centers

    %% Accuracy/cost sweep
    cfg.tradeoff.orders = [6 8 10];
    cfg.tradeoff.blocksPerDimension = 1:6;
    cfg.tradeoff.mode = 10;
    % Full-size GPU warmups would roughly double the production runtime.
    % GPU kernels are instead checked by the dedicated preflight.
    cfg.tradeoff.warmupEachConfiguration = false;

    %% Workflow timing
    cfg.workflow.mode = 10;
    cfg.workflow.reuseTradeoffTiming = true;

    %% Output / run switches
    cfg.run.solve = true;
    cfg.run.makePlots = false;

    cfg.output.verbose = true;
    cfg.output.pngResolution = 600;
    cfg.output.resultFile = 'Experiment2_N5_n40_AllGPU_results.mat';
    cfg.output.tradeoffCSV = 'Experiment2_N5_Fig2a_tradeoff_GPU.csv';
    cfg.output.workflowCSV = 'Experiment2_N5_Fig2b_workflow_GPU.csv';
    cfg.output.eigensolverBreakdownCSV = 'Experiment2_N5_eigensolver_breakdown_GPU.csv';
    cfg.output.preparationBreakdownCSV = 'Experiment2_N5_preparation_breakdown_GPU.csv';
    cfg.output.performanceCSV = 'Experiment2_N5_reconstruction_performance_GPU.csv';
    cfg.output.summaryTXT = 'Experiment2_N5_AllGPU_summary.txt';
    cfg.output.latexTXT = 'Experiment2_N5_AllGPU_LaTeX_values.txt';
    cfg.output.figureTradeoffBase = 'Figure_Exp2a_OrderCenterTradeoff_N5_GPU';
    cfg.output.figureWorkflowBase = 'Figure_Exp2b_WorkflowTiming_N5_GPU';
end
