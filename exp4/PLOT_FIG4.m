function figs = PLOT_FIG4()
%PLOT_FIG4 Generate exactly the six panels used in Figure 4.
%   Requires Exp4_A_results.mat and Exp4_B_results.mat in output/.
    setup_exp4();
    [A,B] = load_exp4_two_media();

    S = plot_exp4_spectrum_fig4(A,B,true);
    figs = struct();
    figs.spectrumA = S.A;
    figs.fieldA_z0 = plot_exp4_field_slice_group(A,'z0',true);
    figs.fieldA_xy = plot_exp4_field_slice_group(A,'xy',true);
    figs.spectrumB = S.B;
    figs.fieldB_z0 = plot_exp4_field_slice_group(B,'z0',true);
    figs.fieldB_xy = plot_exp4_field_slice_group(B,'xy',true);

    fprintf('Figure 4 complete: six panels written to %s\n',fullfile(setup_exp4(),'figure'));
end
