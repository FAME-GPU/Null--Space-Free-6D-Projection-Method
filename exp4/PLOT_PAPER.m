function PLOT_PAPER()
%PLOT_PAPER Three panels for the manuscript equation (6.2) medium.
    R=load_exp4_results('PAPER');
    assert(isfield(R,'version') && strcmp(R.version,'Exp4_Public_v2_ElementwiseRitz_AxialMaterial'));
    assert(all(isfinite(R.solverStats.maxRelativeInverseRitzResidual)) && ...
        max(R.solverStats.maxRelativeInverseRitzResidual)<=R.cfg.solver.tolLanczos, ...
        'Spectrum fails the requested Ritz residual tolerance.');
    plot_exp4_spectrum_single(R,true);
    plot_exp4_field_slice_group(R,'z0',true);
    plot_exp4_field_slice_group(R,'xy',true);
end
