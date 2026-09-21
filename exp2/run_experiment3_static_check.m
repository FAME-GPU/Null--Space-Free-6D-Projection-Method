function run_experiment3_static_check(cfg)
%RUN_EXPERIMENT3_STATIC_CHECK Cheap configuration validation, no GPU solve.

    assert(cfg.problem.N==5,'Production N must be 5.');
    assert(cfg.yee.n==40,'Yee n must be 40.');
    assert(abs(cfg.yee.h-0.0125)<1e-15,'Yee h must be 0.0125.');
    assert(abs(cfg.yee.n*cfg.yee.h-0.5)<1e-14,'n*h must equal 0.5.');
    assert(strcmpi(cfg.mass.backend,'rank1-exact-scalar'), ...
        'Production mass backend must be rank1-exact-scalar.');

    NF=(2*cfg.problem.N)^cfg.problem.dim;
    M=cfg.mass.rank1.N5.M;
    z=cfg.mass.rank1.N5.z(:).';
    expectedM=24760990;
    expectedZ=[1 19 361 6859 130321 2476099];
    assert(NF==1000000,'Expected NF=1,000,000.');
    assert(M==expectedM,'Unexpected N=5 rank-1 M.');
    assert(isequal(double(z),double(expectedZ)),'Unexpected N=5 rank-1 z.');

    Ng=(2*cfg.yee.n+3)^3;
    directPeakGiB=(8+16)*double(cfg.direct.blockSize)*double(NF)/2^30;
    denseGiB=double(Ng)*double(NF)*16/2^30;

    required={ ...
        'Lanczos','build_radix_exact_rank1_plan_scalar', ...
        'build_exact_rank1_scalar_kernel_fft','applyM_component_first_rank1_scalar', ...
        'make_krinv_instrumented','recover_original_eigenvectors_exp2', ...
        'reconstruct_three_components_exp3','plot_experiment2a_tradeoff'};
    for j=1:numel(required)
        if exist(required{j},'file')~=2
            error('Missing required function: %s',required{j});
        end
    end



    recoveryPath=which('recover_original_eigenvectors_exp2');
    recoveryText=fileread(recoveryPath);
    badRecoveryLine=regexp(recoveryText, ...
        '(?m)^\s*MX\s*=\s*sys\.mass\.Mop\(X\)\s*;','once');
    if ~isempty(badRecoveryLine)
        error(['Recovery implementation still contains the N=5-unsafe ', ...
            'multi-RHS mass action.']);
    end
    if ~contains(recoveryText,'memorySafeColumnwiseMassAction')
        error('Memory-safe columnwise recovery marker not found.');
    end

    fprintf('[Exp3 static check] PASS\n');
    fprintf('  N=%d, NF=%d, reduced dim=%d\n',cfg.problem.N,NF,2*NF);
    fprintf('  rank-1 M=%d, M/NF=%.6f\n',M,M/NF);
    fprintf('  Yee n=%d, h=%.5f, Ng=%d, physical width=%.3f\n', ...
        cfg.yee.n,cfg.yee.h,Ng,2*cfg.yee.n*cfg.yee.h);
    fprintf('  direct block estimated auxiliary workspace = %.2f GiB\n',directPeakGiB);
    fprintf('  full dense phase matrix per component = %.2f GiB\n',denseGiB);
    fprintf('  recovery mass actions: columnwise memory-safe mode PASS\n');
end
