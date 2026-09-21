function L = compute_lwrq_exp4(a,b,h,cfg,returnRho)
%COMPUTE_LWRQ_EXP4 GPU-safe column-normalized local Gaussian RQ.
%
% Only scalar diagnostics are retained by default.  The full 3D rho field is
% kept only when a plotting slice is explicitly requested (the L=4 case).

    if nargin<5, returnRho=false; end
    if ~isa(a,'gpuArray') || ~isa(b,'gpuArray')
        error('All-GPU Exp3 requires a and b to be gpuArray.');
    end

    R=cfg.lwrq.neighborRadiusCells;
    sigma=cfg.lwrq.sigmaFactor*h;
    off=(-R:R)*h;
    k=cast(exp(-(off.^2)/(sigma^2)),'like',a);

    den=separable_conv3_same_exp4(ones(size(a),'like',a),k);
    badDen=gather_scalar(any(~isfinite(den(:)) | den(:)<=0));
    if badDen, error('Invalid LWRQ column-normalization denominator.'); end

    aNorm=a./den;
    E=separable_conv3_same_exp4(aNorm,k);
    clear aNorm
    bNorm=b./den;
    M=separable_conv3_same_exp4(bNorm,k);
    clear bNorm den

    maxM=gather_scalar(max(M(:)));
    massFloor=cfg.lwrq.massFloorRelative*maxM;
    badM=gather_scalar(any(~isfinite(M(:)) | M(:)<=massFloor));
    if badM, error('Nonpositive or numerically tiny local LWRQ mass encountered.'); end

    sumE=sum(E(:));
    sumM=sum(M(:));
    lambda=sumE/sumM;
    rho=E./M;
    Vnum=sum(M(:).*abs(rho(:)-lambda).^2);
    V=Vnum/(sumM*abs(lambda)^2);

    sumA=sum(a(:)); sumB=sum(b(:));
    tinyA=cast(eps,'like',real(sumA));
    tinyB=cast(eps,'like',real(sumB));
    energyPartitionError=abs(sumE-sumA)/max(abs(sumA),tinyA);
    massPartitionError=abs(sumM-sumB)/max(abs(sumB),tinyB);
    deltaPart=max(energyPartitionError,massPartitionError);

    L=struct();
    L.lambda=lambda;
    L.V=real(V);
    L.nu=sqrt(max(real(V),0));
    L.partitionError=deltaPart;
    L.energyPartitionError=energyPartitionError;
    L.massPartitionError=massPartitionError;
    if returnRho
        L.rho=rho;
    else
        L.rho=[];
    end
    clear E M rho
end
