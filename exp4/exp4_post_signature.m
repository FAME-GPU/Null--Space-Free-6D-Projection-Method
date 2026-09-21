function s = exp4_post_signature(cfg,j,lambda6D)
%EXP4_POST_SIGNATURE Compatibility signature for selected postprocessing.
    s=struct();
    s.numerics=struct('problem',cfg.problem,'solver',cfg.solver,'mass',cfg.mass, ...
        'taylor',cfg.taylor,'lwrq',cfg.lwrq,'visual',cfg.visual);
    s.version=6;
    s.medium=cfg.material.name;
    s.modelType=cfg.material.modelType;
    s.materialParameters=material_parameter_vector(cfg.material);
    s.gpuProduction=true;
    s.N=cfg.problem.N;
    s.pathSubintervals=cfg.path.subintervalsPerSegment;
    s.point=cfg.selected.pointIndices(j);
    s.mode=cfg.selected.modeIndices(j);
    s.lambda6D=lambda6D;
    s.taylorOrder=cfg.taylor.order;
    s.phaseRadius=cfg.taylor.phaseRadiusTarget;
    s.lwrqH=cfg.lwrq.h;
    s.lwrqN=cfg.lwrq.n;
    s.visualHalfWidth=cfg.visual.halfWidth;
    s.visualNumPoints=cfg.visual.numPoints;
    s.visualPhaseRadius=cfg.visual.phaseRadiusTarget;
    s.visualPlanes=cfg.visual.planes;
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
