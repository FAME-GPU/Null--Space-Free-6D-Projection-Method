function L = compute_lwrq_exp4(a,b,h,cfg)
%COMPUTE_LWRQ_EXP4 GPU column-normalized local Gaussian Rayleigh quotients.
    if ~isa(a,'gpuArray') || ~isa(b,'gpuArray')
        error('LWRQ production path requires GPU arrays.');
    end

    R=cfg.lwrq.neighborRadiusCells;
    sigma=cfg.lwrq.sigmaFactor*h;
    offCPU=(-R:R)*h;
    off=gpuArray(cast(offCPU,classUnderlying(a)));
    sigmaGPU=cast(sigma,'like',a);
    k=exp(-(off.^2)/(sigmaGPU^2));

    den=separable_conv3_same_exp4(ones(size(a),'like',a),k);
    if gather_scalar(any(~isfinite(den(:)))) || gather_scalar(any(den(:)<=0))
        error('Invalid LWRQ column-normalization denominator.');
    end

    E=separable_conv3_same_exp4(a./den,k);
    M=separable_conv3_same_exp4(b./den,k);
    massFloor=cast(cfg.lwrq.massFloorRelative,'like',M).*max(M(:));
    if gather_scalar(any(~isfinite(M(:)))) || gather_scalar(any(M(:)<=massFloor))
        error('Nonpositive or numerically tiny local LWRQ mass encountered.');
    end

    rho=E./M;
    p=M./sum(M(:));
    lambda=sum(p(:).*rho(:));
    V=sum(p(:).*abs(rho(:)-lambda).^2)./(abs(lambda).^2);

    tiny=cast(eps('double'),'like',a);
    energyPartitionError=abs(sum(E(:))-sum(a(:)))./max(abs(sum(a(:))),tiny);
    massPartitionError=abs(sum(M(:))-sum(b(:)))./max(abs(sum(b(:))),tiny);
    deltaPart=max(energyPartitionError,massPartitionError);

    L=struct();
    L.E=E; L.M=M; L.rho=rho; L.p=p;
    L.lambda=lambda;
    L.V=real(V);
    L.nu=sqrt(max(real(V),cast(0,'like',V)));
    L.partitionError=deltaPart;
    L.energyPartitionError=energyPartitionError;
    L.massPartitionError=massPartitionError;
    L.kernel1D=k;
end
