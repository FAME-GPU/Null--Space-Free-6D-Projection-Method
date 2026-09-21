function cfg = make_config_BC_fig1a()
%MAKE_CONFIG_BC_FIG1A Frozen Method-B/C configuration for Exp. 1 Fig. 1(a).
%
% This file intentionally contains ONLY the settings needed by Methods B/C
% at N=5 along the 12-point Bloch path.  The redesigned Fig. 1(b,c)
% scalability experiment is not part of this package.

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

    cfg.lanczos.krylovDimension = 40;
    cfg.lanczos.tolerance = 1e-10;
    cfg.lanczos.maxRestartCycles = 50;
    cfg.lanczos.verboseInternal = false;
    cfg.lanczos.rngSeed = 20260905;

    cfg.nested.tolKr = 1e-10;
    cfg.nested.tolM = 1e-12;
    cfg.nested.maxitKr = 600;
    cfg.nested.maxitM = 1600;
    cfg.nested.useKrPreconditioner = true;

    cfg.explicit.tolMhat = 1e-10;
    cfg.explicit.maxitMhat = 1200;

    % Fig. 1(a) is an accuracy/work comparison, not a scalability-budget
    % run.  No finite iteration/time cutoff is imposed here.
    cfg.output.directory = 'results';
    cfg.output.checkpointDirectory = 'checkpoint';
end
