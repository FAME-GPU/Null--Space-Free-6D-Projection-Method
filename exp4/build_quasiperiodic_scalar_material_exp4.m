function [mat,info] = build_quasiperiodic_scalar_material_exp4(materialCfg,x1,x2,x3,x4,x5,x6)
%BUILD_QUASIPERIODIC_SCALAR_MATERIAL_EXP4 Build Experiment-4 medium on GPU.
    if ~isa(x1,'gpuArray')
        error('Experiment 4 production material construction must run on the GPU.');
    end

    stats=exp4_material_stats_gpu(materialCfg,x1);

    e=evaluate_exp4_permittivity(materialCfg,x1,x2,x3,x4,x5,x6);
    if strcmp(materialCfg.modelType,'reference_A')
        epsScale=cast(materialCfg.referenceEpsC,'like',x1); rawMean=stats.meanExpA;
    else
        epsScale=cast(materialCfg.cB,'like',x1); rawMean=cast(materialCfg.b0,'like',x1);
    end

    mat=struct('scalarField',e);
    info=struct();
    info.modelType=materialCfg.modelType;
    info.meanQuadratureN=materialCfg.meanQuadratureN;
    info.meanExpA=stats.meanExpA;
    info.meanFactorB=stats.meanFactorB;
    info.referenceMeanPermittivity=stats.referenceMeanPermittivity;
    info.epsScale=epsScale;
    info.rawMeanFactor=rawMean;
    info.targetMeanPermittivity=stats.referenceMeanPermittivity;
    info.continuumMeanPermittivity=epsScale.*rawMean;
    info.relativeContinuumMeanMismatch=abs(info.continuumMeanPermittivity-info.targetMeanPermittivity) ...
        ./abs(info.targetMeanPermittivity);
end
