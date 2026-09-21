function run_experiment2_allgpu_static_check(cfg)
%RUN_EXPERIMENT2_ALLGPU_STATIC_CHECK Cheap no-GPU configuration/source audit.
    assert(cfg.problem.N==5); assert(cfg.yee.n==40); assert(abs(cfg.yee.h-0.0125)<1e-15);
    assert(cfg.gpu.useGPU,'GPU execution must be enabled.');
    assert(cfg.direct.blockSize==256,'Final direct block size must be 256.');
    assert(isequal(cfg.tradeoff.orders,[6 8 10]));
    assert(isequal(cfg.tradeoff.blocksPerDimension,1:6));
    assert(cfg.taylor.performanceOrder==10 && cfg.taylor.blocksPerDimension==4);
    NF=(2*cfg.problem.N)^cfg.problem.dim; assert(NF==1000000);
    assert(cfg.mass.rank1.N5.M==24760990);
    assert(isequal(double(cfg.mass.rank1.N5.z(:).'),double([1 19 361 6859 130321 2476099])));

    required={'run_experiment2_allgpu','plot_experiment2a_tradeoff','reconstruct_three_components_exp3', ...
        'reconstruct_field_direct_blocked_exp3','taylor_on_blocks_exp3','apply_cropped_A_exp3', ...
        'compute_field_errors_exp3','compute_A_error_exp3','solve_modes_exp3'};
    for j=1:numel(required), if exist(required{j},'file')~=2, error('Missing required function: %s',required{j}); end, end

    txt=fileread(which('run_experiment2_allgpu'));
    prohibited={'to_cpu_yee_points_exp3','XModeCPU','qModesCPU','yeeCPU','reconstructionOnCPU','CPUTimeSeconds'};
    for j=1:numel(prohibited)
        if contains(txt,prohibited{j}), error('All-GPU runner still contains prohibited legacy token: %s',prohibited{j}); end
    end
    rec=fileread(which('reconstruct_three_components_exp3'));
    if contains(rec,'gather(') || contains(rec,'to_cpu('), error('Reconstruction wrapper contains host transfer.'); end

    directPeakGiB=(8+16)*double(cfg.direct.blockSize)*double(NF)/2^30;
    Ng=(2*cfg.yee.n+3)^3; denseGiB=double(Ng)*double(NF)*16/2^30;
    fprintf('[Exp2 all-GPU static check] PASS\n');
    fprintf('  N=5, NF=%d, reduced dim=%d, Ng=%d\n',NF,2*NF,Ng);
    fprintf('  direct block=%d, estimated phase workspace=%.2f GiB\n',cfg.direct.blockSize,directPeakGiB);
    fprintf('  dense phase matrix/component=%.2f GiB\n',denseGiB);
    fprintf('  source audit: no legacy CPU reconstruction transfers in production runner.\n');
end
