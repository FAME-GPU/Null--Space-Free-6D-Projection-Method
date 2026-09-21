function sig = make_A_fig1a_signature(cfg,gpuInfo)
%MAKE_A_FIG1A_SIGNATURE Checkpoint compatibility signature.
    sig = struct();
    sig.version = 2;
    sig.experiment = 'Exp1-Fig1a-A-only-plateau';
    sig.N = cfg.problem.N;
    sig.P = cfg.problem.P;
    sig.T = cfg.problem.T;
    sig.material = cfg.material;
    sig.massBackend = cfg.mass.backend;
    sig.pathVertices = cfg.path.vertices;
    sig.pathInteriorFractions = cfg.path.interiorFractions;
    sig.numEig = cfg.spectrum.numEig;
    sig.eigs = cfg.eigs;
    sig.shift = cfg.shift;
    sig.precision = cfg.gpu.precision;
    sig.gpuName = gpuInfo.name;
    sig.matlabVersion = version;
end
