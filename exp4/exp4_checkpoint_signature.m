function sig = exp4_checkpoint_signature(cfg,q,pointIndex,selectedMode)
%EXP4_CHECKPOINT_SIGNATURE Configuration subset that makes q checkpoints valid.
    if nargin<4 || isempty(selectedMode)
        j=find(cfg.selected.pointIndices==pointIndex,1);
        if isempty(j), selectedMode=0; else, selectedMode=cfg.selected.modeIndices(j); end
    end
    sig=struct();
    sig.numerics=struct('problem',cfg.problem,'solver',cfg.solver,'mass',cfg.mass);
    sig.version=6;
    sig.medium=cfg.material.name;
    sig.modelType=cfg.material.modelType;
    sig.materialParameters=material_parameter_vector(cfg.material);
    sig.gpuProduction=true;
    sig.N=cfg.problem.N;
    sig.pathSubintervals=cfg.path.subintervalsPerSegment;
    sig.nev=cfg.spectrum.nev;
    sig.p=cfg.spectrum.p;
    sig.tolLanczos=cfg.solver.tolLanczos;
    sig.tolCG=cfg.solver.tolCG;
    sig.pointIndex=pointIndex;
    sig.q=q(:).';
    sig.selectedMode=selectedMode;
end

function v=material_parameter_vector(mat)
    switch char(mat.modelType)
        case 'reference_A'
            v=[mat.referenceEpsC mat.alpha mat.beta mat.meanQuadratureN];
        case 'mixed_fourier_B'
            v=[mat.referenceEpsC mat.cB mat.b0 mat.a12 mat.a34 mat.a56 mat.a135 mat.axialAmplitude mat.meanQuadratureN];
        otherwise
            error('Unknown material modelType: %s',mat.modelType);
    end
end
