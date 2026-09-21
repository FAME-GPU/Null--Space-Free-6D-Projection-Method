function mat = build_smooth_scalar_material_compact(materialCfg,x1,x2,x3,x4,x5,x6)
%BUILD_SMOOTH_SCALAR_MATERIAL_COMPACT Scalar heterogeneous permittivity only.
    epsC=materialCfg.epsC;
    alpha=materialCfg.alpha;
    phi=(alpha/6)*(cos(x1)+cos(x2)+cos(x3)+cos(x4)+cos(x5)+cos(x6));
    e=epsC*exp(phi);
    mat=struct('scalarField',e);
end
