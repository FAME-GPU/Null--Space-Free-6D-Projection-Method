function run_experiment2_reconstruction_timing_pilot(cfg,rootDir)
%RUN_EXPERIMENT2_RECONSTRUCTION_TIMING_PILOT Full-grid one-component GPU timing pilot.
% Uses a synthetic Fourier coefficient vector; no eigensolve is required.
    if nargin<2, rootDir=fileparts(mfilename('fullpath')); end %#ok<NASGU>
    gpuInfo=gpu_initialize(cfg.gpu);
    sys=build_experiment3_system(cfg,5,gpuInfo);
    qModes=build_projected_q_modes(sys.P,sys.kLift,sys.fourier.Xi,sys.NF,cfg.problem.dim);
    yee=build_yee_points_exp3(cfg.yee.n,cfg.yee.h,sys.P(1));
    rng(cfg.solver.rngSeed+99,'twister');
    coeff=to_gpu(randn(sys.NF,1)+1i*randn(sys.NF,1),gpuInfo.precision);
    fprintf('[Exp2 timing pilot] full-grid one-component direct...\n');
    [uD,iD]=reconstruct_field_direct_blocked_exp3(coeff,qModes,yee.r1,cfg.direct.blockSize); clear uD
    blocks=build_yee_block_partition_exp3(yee.szPlus,cfg.taylor.blocksPerDimension);
    fprintf('[Exp2 timing pilot] full-grid one-component p=10/64-center Taylor...\n');
    [uT,iT]=reconstruct_field_taylor_multi_exp3(coeff,qModes,yee.r1,cfg.taylor.performanceOrder,cfg.taylor.termBatchSize,blocks); clear uT
    fprintf('  one component direct = %.3f s\n',iD.timeSeconds);
    fprintf('  one component Taylor = %.3f s\n',iT.timeSeconds);
    fprintf('  projected three-component direct/Taylor = %.2f / %.2f min\n',3*iD.timeSeconds/60,3*iT.timeSeconds/60);
    fprintf('  NOTE: production also runs 18 Taylor configurations; use this only as a scale estimate.\n');
end
