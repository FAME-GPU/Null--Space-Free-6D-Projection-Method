function self_check_exp1bc()
%SELF_CHECK_EXP1BC Static/configuration checks before long production runs.
    cfg=make_config_exp1bc();
    assert(isequal(cfg.scale.NList,3:7));
    assert(cfg.budget.maxTotalLinearIterations==1e6);
    assert(cfg.budget.maxSolverWallSeconds==1e4);
    assert(cfg.shift.minresTolerance==1e-10);
    assert(cfg.shift.plateauAcceptanceCeiling==1e-9);
    assert(cfg.shift.guardNearToleranceCeiling==1e-9);
    assert(cfg.shift.guardNearToleranceCeiling>=cfg.shift.minresTolerance);
    assert(cfg.shift.guardNearToleranceCeiling<=cfg.shift.plateauAcceptanceCeiling);
    assert(cfg.lanczos.hybridCPUOuterBasis);
    assert(cfg.explicit.useScalarCompactPath);
    assert(strcmp(cfg.mass.backend,'rank1-exact-scalar-lowmem'));

    % Checkpoint compatibility policy: Method A is invalidated by the
    % final 1e-9 hard-guard plateau rule; B/C remain on frozen v4 signatures.
    sigA=make_exp1bc_signature(cfg,'A',3);
    sigB=make_exp1bc_signature(cfg,'B',3);
    sigC=make_exp1bc_signature(cfg,'C',3);
    assert(sigA.version==6 && isfield(sigA,'guardNearToleranceCeiling'));
    assert(sigB.version==4 && ~isfield(sigB,'guardNearToleranceCeiling'));
    assert(sigC.version==4 && ~isfield(sigC,'guardNearToleranceCeiling'));

    % Deterministic regression checks for the final Method-A hard guards.
    tolA=cfg.shift.minresTolerance;
    guardA=cfg.shift.guardNearToleranceCeiling;
    plateauA=cfg.shift.plateauAcceptanceCeiling;
    minEvents=cfg.shift.plateauMinFailedRecoveryCycles;
    ordinaryPlateau=@(r,events) (events>=minEvents && isfinite(r) && r<=plateauA);
    hardGuardAccept=@(r) (isfinite(r) && r<=guardA);
    assert(hardGuardAccept(1.152e-10));  % previous observed failure
    assert(hardGuardAccept(2.021e-10));  % residual-correction-guard failure reported later
    assert(hardGuardAccept(9.99e-10));
    assert(~hardGuardAccept(1.001e-9));
    assert(ordinaryPlateau(8e-10,minEvents));
    assert(~ordinaryPlateau(8e-10,0));
    assert(hardGuardAccept(tolA));
    assert(~hardGuardAccept(Inf) && ~hardGuardAccept(NaN));

    root=exp1bc_project_root();
    requiredDirs={cfg.output.resultsDirectory,cfg.output.checkpointDirectory,cfg.output.figureDirectory};
    for id=1:numel(requiredDirs)
        d=fullfile(root,requiredDirs{id});
        assert(exist(d,'dir')==7,'Missing required output directory: %s',d);
    end
    assert(isfield(cfg.output,'figureBBase') && isfield(cfg.output,'figureCBase'));
    assert(cfg.output.pngResolution>=300);

    fprintf('N   NF           rank1 M       kernel GiB   CPU B/C basis GiB\n');
    for N=cfg.scale.NList
        M=(2*N)*(4*N-1)^5;
        assert(M<double(intmax('uint32')));
        nf=(2*N)^6; kernelGiB=M*16/2^30;
        basisGiB=(2*nf)*cfg.lanczos.krylovDimension*16/2^30;
        fprintf('%d   %-12.0f %-13.0f %.3f        %.3f\n',N,nf,M,kernelGiB,basisGiB);
    end

    required={ ...
        'run_fig1bc_method.m','run_method_A_fig1bc.m','run_method_B_fig1bc.m','run_method_C_fig1bc.m', ...
        'make_A_eigs_top_preconditioned.m','make_nested_krinv_instrumented.m','make_explicit_krinv_compact.m', ...
        'Lanczos_hybrid_identity_gpuop.m','build_experiment1_q_system_C_compact.m', ...
        'applyMij_rank1_exact_lowmem.m','classify_exp1_exception.m'};
    for i=1:numel(required)
        assert(~isempty(which(required{i})),'Missing %s on MATLAB path',required{i});
    end
    fprintf('Static/configuration self-check PASSED.\n');
    fprintf('Run START_SMOKE_GPU before the long N=3:7 jobs.\n');
end
