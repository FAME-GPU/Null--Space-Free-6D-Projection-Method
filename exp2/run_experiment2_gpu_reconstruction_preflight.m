function run_experiment2_gpu_reconstruction_preflight(cfg,rootDir)
%RUN_EXPERIMENT2_GPU_RECONSTRUCTION_PREFLIGHT Validate actual GPU reconstruction kernels cheaply.
    if nargin<2, rootDir=fileparts(mfilename('fullpath')); end %#ok<NASGU>
    gpuInfo=gpu_initialize(cfg.gpu);
    fprintf('[Exp2 GPU reconstruction preflight] building N=5 system...\n');
    sys=build_experiment3_system(cfg,5,gpuInfo);
    qModes=build_projected_q_modes(sys.P,sys.kLift,sys.fourier.Xi,sys.NF,cfg.problem.dim);
    if ~isa(qModes,'gpuArray'), error('qModes not on GPU.'); end
    rng(cfg.solver.rngSeed,'twister');
    coeff=to_gpu(randn(sys.NF,1)+1i*randn(sys.NF,1),gpuInfo.precision);
    % Only 96 physical points are used here; this validates code paths without a full reconstruction.
    proto=sys.P(1);
    pts=cast(linspace(-0.5,0.5,96).','like',proto);
    r=[pts,0*pts,0*pts];
    fprintf('  direct kernel on 96 GPU points...\n');
    [uD,infoD]=reconstruct_field_direct_blocked_exp3(coeff,qModes,r,32);
    if ~isa(uD,'gpuArray') || any(~isfinite(gather(uD(1:min(4,end))))), error('GPU direct kernel failed.'); end
    blocks={1:size(r,1)};
    fprintf('  Taylor kernel p=6 on same GPU points...\n');
    [uT,infoT]=reconstruct_field_taylor_multi_exp3(coeff,qModes,r,6,16,blocks);
    if ~isa(uT,'gpuArray') || any(~isfinite(gather(uT(1:min(4,end))))), error('GPU Taylor kernel failed.'); end
    fprintf('  direct %.3f s, Taylor %.3f s, Taylor rho %.4f\n',infoD.timeSeconds,infoT.timeSeconds,infoT.rhoMax);
    fprintf('[Exp2 GPU reconstruction preflight] PASS\n');
end
