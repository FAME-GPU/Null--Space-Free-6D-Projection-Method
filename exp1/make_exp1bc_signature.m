function s = make_exp1bc_signature(cfg,method,N)
%MAKE_EXP1BC_SIGNATURE Complete checkpoint compatibility signature.
    s=struct();
    method=upper(char(method));
    if method=='A'
        % Version 6 invalidates older Method-A checkpoints because the
        % hard-guard plateau acceptance ceiling is now 1e-9.  Methods B/C keep
        % version 4 so their existing checkpoints remain reusable.
        s.version=6;
    else
        s.version=4;
    end
    s.kind='fig1bc-hybrid-lowmem';
    s.method=method; s.N=N;
    s.numEig=cfg.spectrum.numEig; s.P=cfg.problem.P; s.T=cfg.problem.T;
    s.epsC=cfg.material.epsC; s.alpha=cfg.material.alpha;
    s.rank1Backend=cfg.mass.backend; s.q=cfg.scale.qPhys; s.NList=cfg.scale.NList;
    s.eigsTol=cfg.eigs.tolerance; s.eigsMaxIterations=cfg.eigs.maxIterations;
    s.eigsP=cfg.eigs.subspaceDimension; s.eigsSeed=cfg.eigs.rngSeed;
    s.sigma=cfg.shift.sigma; s.minresTol=cfg.shift.minresTolerance;
    s.minresMaxit=cfg.shift.minresMaxit; s.useSpectralPreconditioner=cfg.shift.useSpectralPreconditioner;
    s.maxRecoveryEvents=cfg.shift.maxRecoveryEventsPerTop;
    s.maxCorrectionCycles=cfg.shift.maxResidualCorrectionCycles;
    s.maxZeroProgressEvents=cfg.shift.maxZeroProgressEvents;
    s.minResidualProgressFraction=cfg.shift.minResidualProgressFraction;
    s.correctionSafetyFactor=cfg.shift.correctionSafetyFactor;
    s.correctionMaxTol=cfg.shift.correctionMaxRelativeTolerance;
    s.correctionMinTol=cfg.shift.correctionMinRelativeTolerance;
    s.plateauCeiling=cfg.shift.plateauAcceptanceCeiling;
    s.plateauMinCycles=cfg.shift.plateauMinFailedRecoveryCycles;
    s.plateauMinReduction=cfg.shift.plateauMinRelativeReduction;
    if method=='A'
        s.guardNearToleranceCeiling=cfg.shift.guardNearToleranceCeiling;
    end
    s.lanczosTol=cfg.lanczos.tolerance; s.lanczosMaxRestart=cfg.lanczos.maxRestartCycles;
    s.lanczosP=cfg.lanczos.krylovDimension; s.lanczosSeed=cfg.lanczos.rngSeed;
    s.hybridCPUOuterBasis=cfg.lanczos.hybridCPUOuterBasis;
    s.tolKr=cfg.nested.tolKr; s.tolM=cfg.nested.tolM;
    s.maxitKr=cfg.nested.maxitKr; s.maxitM=cfg.nested.maxitM;
    s.useKrPreconditioner=cfg.nested.useKrPreconditioner;
    s.tolMhat=cfg.explicit.tolMhat; s.maxitMhat=cfg.explicit.maxitMhat;
    s.scalarCompactC=cfg.explicit.useScalarCompactPath;
    s.iterBudget=cfg.budget.maxTotalLinearIterations;
    s.timeBudget=cfg.budget.maxSolverWallSeconds;
    s.resetBetweenN=cfg.gpu.resetBetweenN;
end
