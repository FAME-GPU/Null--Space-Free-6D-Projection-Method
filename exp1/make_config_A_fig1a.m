function cfg = make_config_A_fig1a()
%MAKE_CONFIG_A_FIG1A Formal A-only configuration for Experiment 1 Fig. 1(a).
%
% Purpose
% -------
% Recompute Method A only at N=5 along the 12 independent Bloch points.
%
% Frozen Method A
% ---------------
%   original GEVP: K*u = lambda*M*u
%   outer          : MATLAB eigs on CPU
%   transformed op : (K-sigma*M)^(-1) M
%   target          : largestreal
%   inner           : MATLAB MINRES on GPU
%   preconditioner  : exact modewise inverse of K+sigma*mean(eps)*I
%
% The requested MINRES tolerance remains 1e-10.  Plateau acceptance does
% NOT change this target.  It is a fallback used only after repeated
% recovery cycles make no material progress and the measured true residual
% is already below 1e-9.

    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.precision = 'double';

    cfg.problem.dim = 6;
    cfg.problem.T = 2*pi*ones(6,1);
    cfg.problem.N = 5;
    cfg.problem.P = [ ...
        1       0       0; ...
        0       1       0; ...
        0       0       1; ...
        sqrt(2) 0       0; ...
        0       sqrt(3) 0; ...
        0       0       sqrt(5)];

    cfg.material.epsC = 4;
    cfg.material.alpha = 1.2;

    cfg.mass.backend = 'rank1-exact-scalar';
    cfg.mass.rank1 = struct();

    cfg.path.vertices = [ ...
        0.05 0.25 0.25 0.25 0.05; ...
        0.04 0.04 0.20 0.20 0.04; ...
        0.03 0.03 0.03 0.18 0.03];
    cfg.path.interiorFractions = [1/3 2/3];

    cfg.spectrum.numEig = 10;

    % MATLAB eigs settings.
    cfg.eigs.tolerance = 1e-10;
    cfg.eigs.maxIterations = 300;
    cfg.eigs.subspaceDimension = 40;
    cfg.eigs.rngSeed = 20260906;
    cfg.eigs.display = false;

    % Shifted MINRES settings.
    cfg.shift.sigma = 1e-4;
    cfg.shift.minresTolerance = 1e-10;
    cfg.shift.minresMaxit = 4000;
    cfg.shift.useSpectralPreconditioner = true;
    cfg.shift.preconditionerTauMode = 'sigma-times-mean-epsilon';

    % Existing bounded recovery logic.
    cfg.shift.maxRecoveryEventsPerTop = 50;
    cfg.shift.maxResidualCorrectionCycles = 8;
    cfg.shift.maxZeroProgressEvents = 2;
    cfg.shift.minResidualProgressFraction = 1e-8;
    cfg.shift.correctionSafetyFactor = 0.25;
    cfg.shift.correctionMaxRelativeTolerance = 1e-1;
    cfg.shift.correctionMinRelativeTolerance = 1e-6;

    % New plateau fallback.  A recovery/restart cycle must reduce the true
    % residual by at least 1 percent to be considered material progress for
    % plateau detection.  Two consecutive non-material recovery cycles are
    % required before a residual below 1e-9 may be accepted.
    cfg.shift.plateauAcceptanceCeiling = 1e-9;
    cfg.shift.plateauMinFailedRecoveryCycles = 2;
    cfg.shift.plateauMinRelativeReduction = 1e-2;

    cfg.runtime.verboseInner = false;

    % Final original-GEVP diagnostics are outside the eigs timing.
    cfg.diagnostics.computeOriginalGEVPResidual = true;
    cfg.diagnostics.gevpResidualWarning = 1e-8;

    cfg.output.directory = 'results';
    cfg.output.checkpointDirectory = 'checkpoint';
    cfg.output.matFile = 'fig1a_A.mat';
    cfg.output.csvFile = 'fig1a_A.csv';
    cfg.output.summaryFile = 'fig1a_A_summary.txt';
end
