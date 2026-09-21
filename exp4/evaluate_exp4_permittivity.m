function e = evaluate_exp4_permittivity(mat,x1,x2,x3,x4,x5,x6)
%EVALUATE_EXP4_PERMITTIVITY Shared 6D and physical Yee material definition.
% Supports CPU scalars/arrays for tests and gpuArray for production.
    axial = cos(x1)+cos(x2)+cos(x3)+cos(x4)+cos(x5)+cos(x6);
    switch char(mat.modelType)
        case 'reference_A'
            mixed=cos(x1+x4)+cos(x2+x5)+cos(x3+x6);
            e=mat.referenceEpsC.*exp((mat.alpha/6).*axial+(mat.beta/3).*mixed);
        case 'mixed_fourier_B'
            c14=cos(x1-x4); c25=cos(x2-x5); c36=cos(x3-x6);
            a=0;
            if isfield(mat,'axialAmplitude'), a=mat.axialAmplitude; end
            e=mat.cB.*(mat.b0+mat.a12.*c14.*c25+mat.a34.*c25.*c36 ...
                +mat.a56.*c36.*c14+a.*axial);
        otherwise
            error('Unknown material modelType: %s',mat.modelType);
    end
end
