function post = postprocess_selected_exp4(st,cfg,qres,selectedLabel)
%POSTPROCESS_SELECTED_EXP4 GPU LWRQ closure and three visualization planes.
    if isempty(qres.Xselected) || qres.selectedMode<=0
        error('Selected point checkpoint does not contain its Fourier eigenvector.');
    end

    qCPU=qres.q(:);
    qGPU=to_gpu(qCPU,st.gpuInfo.precision);
    Xgpu=gpuArray(cast(qres.Xselected,st.gpuInfo.precision));
    kLift=st.P*((st.P.'*st.P)\qGPU);
    qModes=build_projected_q_modes(st.P,kLift,st.fourier.Xi,st.NF,cfg.problem.dim);

    wait_for_gpu(); tb=tic;
    basis=prepare_taylor_fourier_basis_exp4(qModes,cfg.taylor.order);
    wait_for_gpu(basis.Qalpha); basisSeconds=toc(tb);

    % 3D Yee reconstruction and LWRQ remain on GPU throughout.
    h=cfg.lwrq.h; n=cfg.lwrq.n;
    yee=build_yee_points_exp3(n,h,qModes(1));
    part=build_auto_taylor_partition_exp4(yee,basis,cfg);
    [fields,reconInfo]=reconstruct_tracked_modes_exp4(Xgpu,basis,yee,part,cfg);
    [metrics,lwrqSpatial]=compute_yee_lwrq_metrics_exp4( ...
        fields{1},qres.selectedLambda,cfg,st,h,n,true);
    clear fields yee part

    % Visualization planes are reconstructed and normalized jointly on GPU.
    planes=struct();
    rawMax=cast(0,'like',qModes(1));
    for ip=1:numel(cfg.visual.planes)
        kind=cfg.visual.planes{ip};
        pl=build_visual_plane_exp4(kind,cfg,qModes(1));
        pp=build_visual_plane_partition_exp4(pl,basis,cfg);
        vo=reconstruct_visual_plane_exp4(Xgpu,basis,pl,pp,cfg);
        planes.(kind)=vo;
        rawMax=max(rawMax,max(vo.intensityRaw(:)));
    end
    rawMaxCPU=gather_scalar(rawMax);
    if rawMaxCPU<=0 || ~isfinite(rawMaxCPU), error('Invalid visual-field intensity normalization.'); end
    for ip=1:numel(cfg.visual.planes)
        kind=cfg.visual.planes{ip};
        planes.(kind).intensityNormalized=planes.(kind).intensityRaw./rawMax;
        planes.(kind).normalizationMax=rawMax;
    end

    postGPU=struct();
    postGPU.label=selectedLabel;
    postGPU.pointIndex=qres.pointIndex;
    postGPU.modeIndex=qres.selectedMode;
    postGPU.q=qCPU;
    postGPU.lambda6D=qres.selectedLambda;
    postGPU.lambdaLWRQ=metrics.lambdaLWRQ;
    postGPU.deltaLambda=metrics.deltaLambda;
    postGPU.eta3D=metrics.eta3D;
    postGPU.nuLWRQ=metrics.nuLWRQ;
    postGPU.lwrqIdentityError=metrics.lwrqIdentityError;
    postGPU.lwrqMetrics=metrics;
    postGPU.lwrqSpatial=lwrqSpatial;
    postGPU.reconstructionInfo=reconInfo;
    postGPU.taylorBasisSeconds=basisSeconds;
    postGPU.visual=planes;

    % One recursive gather after every requested calculation has finished.
    post=gather_exp4_for_io(postGPU);
    clear postGPU basis qModes qGPU kLift Xgpu planes metrics lwrqSpatial rawMax
end
