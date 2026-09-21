function cfg = make_config_exp1bc()
%MAKE_CONFIG_EXP1BC Frozen configuration for Experiment 1 Fig. 1(b,c).
%
% One Bloch point q=[0.05,0.04,0.03]^T and N=3:7.
% Fig. 1(b): total iterative linear-solver steps.
% Fig. 1(c): method setup + eigensolve wall time.
%
% Common material/rank-1 construction and GPU reset are excluded from the
% reported wall time.  A/B/C use the same exact scalar rank-1 backend.

    cfg.gpu.useGPU = true;
    cfg.gpu.deviceIndex = 1;
    cfg.gpu.resetDevice = false;
    cfg.gpu.resetBetweenN = true;
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

    cfg.mass.backend = 'rank1-exact-scalar-lowmem';
    cfg.mass.rank1 = struct();

    cfg.scale.qPhys = [0.05;0.04;0.03];
    cfg.scale.NList = 3:7;
    cfg.spectrum.numEig = 10;

    % Method A: CPU MATLAB eigs + GPU preconditioned MINRES.
    cfg.eigs.tolerance = 1e-10;
    cfg.eigs.maxIterations = 300;
    cfg.eigs.subspaceDimension = 40;
    cfg.eigs.rngSeed = 20260906;
    cfg.eigs.display = false;

    cfg.shift.sigma = 1e-4;
    cfg.shift.minresTolerance = 1e-10;
    cfg.shift.minresMaxit = 4000;
    cfg.shift.useSpectralPreconditioner = true;
    cfg.shift.preconditionerTauMode = 'sigma-times-mean-epsilon';
    cfg.shift.maxRecoveryEventsPerTop = 50;
    cfg.shift.maxResidualCorrectionCycles = 8;
    cfg.shift.maxZeroProgressEvents = 2;
    cfg.shift.minResidualProgressFraction = 1e-8;
    cfg.shift.correctionSafetyFactor = 0.25;
    cfg.shift.correctionMaxRelativeTolerance = 1e-1;
    cfg.shift.correctionMinRelativeTolerance = 1e-6;

    % Plateau fallback. The requested MINRES tolerance stays 1e-10.
    % Acceptance is permitted only after repeated no-progress recovery and
    % only if the measured true relative residual is already below 1e-9.
    cfg.shift.plateauAcceptanceCeiling = 1e-9;
    cfg.shift.plateauMinFailedRecoveryCycles = 2;
    cfg.shift.plateauMinRelativeReduction = 1e-2;

    % Hard-guard plateau fallback for Method A only. The requested MINRES
    % tolerance remains 1e-10. If repeated recovery reaches any finite
    % anti-loop guard, the current shifted solve is accepted provided its
    % measured true relative residual is already below 1e-9. This prevents
    % a residual-correction guard from killing the whole eigensolve at a
    % harmless numerical plateau such as 2e-10. Residuals >=1e-9 are not
    % accepted by this fallback.
    cfg.shift.guardNearToleranceCeiling = 1e-9;

    % Methods B/C: restarted inverse Lanczos.  The outer basis is kept on
    % CPU; only each inverse action is transferred to and executed on GPU.
    cfg.lanczos.krylovDimension = 40;
    cfg.lanczos.tolerance = 1e-10;
    cfg.lanczos.maxRestartCycles = 50;
    cfg.lanczos.verboseInternal = false;
    cfg.lanczos.rngSeed = 20260905;
    cfg.lanczos.hybridCPUOuterBasis = true;

    cfg.nested.tolKr = 1e-10;
    cfg.nested.tolM = 1e-12;
    cfg.nested.maxitKr = 600;
    cfg.nested.maxitM = 1600;
    cfg.nested.useKrPreconditioner = true;

    cfg.explicit.tolMhat = 1e-10;
    cfg.explicit.maxitMhat = 1200;
    cfg.explicit.useScalarCompactPath = true;

    % Frozen scalability budgets.
    cfg.budget.maxTotalLinearIterations = 1e6;
    cfg.budget.maxSolverWallSeconds = 1e4;

    cfg.runtime.verboseInner = false;
    cfg.diagnostics.computeAOriginalResidual = false;

    % Output layout: data/checkpoints/figures are separated explicitly.
    cfg.output.resultsDirectory = 'results';
    cfg.output.checkpointDirectory = 'checkpoints';
    cfg.output.figureDirectory = 'figures';
    cfg.output.figureBBase = 'Figure_Exp1b_TotalLinearIterations';
    cfg.output.figureCBase = 'Figure_Exp1c_WallTime';
    cfg.output.pngResolution = 600;
end
