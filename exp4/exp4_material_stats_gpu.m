function out = exp4_material_stats_gpu(materialCfg,prototype)
%EXP4_MATERIAL_STATS_GPU GPU mean/scaling metadata for Experiment 4 media.
%
% Medium A mean is evaluated by periodic 2D quadrature using the pair
% factorization of its exponent. For the new Medium B all nonconstant
% trigonometric terms have zero T^6 mean, hence its exact mean is c_B*b0.

    if nargin<2 || isempty(prototype) || ~isa(prototype,'gpuArray')
        error('exp4_material_stats_gpu requires a gpuArray prototype.');
    end

    nq=materialCfg.meanQuadratureN;
    tCPU=(0:nq-1)*(2*pi/nq);
    t=gpuArray(cast(tCPU,classUnderlying(prototype)));
    [x,y]=ndgrid(t,t);

    aA=cast(1.2,'like',prototype);
    bA=cast(0.3,'like',prototype);
    pairA=(aA/6).*(cos(x)+cos(y)) + (bA/3).*cos(x+y);
    meanExpA=mean(exp(pairA),'all').^3;
    referenceMean=cast(materialCfg.referenceEpsC,'like',prototype).*meanExpA;

    if isfield(materialCfg,'cB'), cB=cast(materialCfg.cB,'like',prototype); else, cB=cast(1.2,'like',prototype); end
    if isfield(materialCfg,'b0'), b0=cast(materialCfg.b0,'like',prototype); else, b0=cast(3.6,'like',prototype); end
    meanFactorB=b0;
    meanB=cB.*b0;

    out=struct();
    out.meanExpA=meanExpA;
    out.referenceMeanPermittivity=referenceMean;
    out.meanFactorB=meanFactorB;
    out.epsB=cB;              % retained field name for downstream metadata compatibility
    out.mediumBMeanPermittivity=meanB;
    clear x y pairA t
end
