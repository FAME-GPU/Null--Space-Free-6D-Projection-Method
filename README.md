Source Code for "Null-Space Free 6D Projection Method with Local Rayleigh Quotient Recovery for 3D Quasiperiodic Maxwell's Eigenproblems"

This repository contains the MATLAB implementation for the paper
"Null-Space Free 6D Projection Method with Local Rayleigh Quotient Recovery for 3D Quasiperiodic Maxwell's Eigenproblems".

The code covers the four numerical experiments in the manuscript, including comparison of original and reduced inverse realizations, three-dimensional Yee-field reconstruction, physical-space consistency and local weighted Rayleigh quotient validation, and spectral and field computations for a representative quasiperiodic Maxwell problem.


1. Requirements

    Software: MATLAB with Parallel Computing Toolbox for numerical computation.
    Reference MATLAB version: MATLAB R2026a.
    GPU: A MATLAB-supported NVIDIA GPU with double-precision support.
    Reference GPU: NVIDIA Tesla V100-SXM2 with 32 GB device memory.
    Host memory: Sufficient RAM for the Fourier and Krylov arrays. At N=7,
        a complex-double 2*NF-by-40 basis alone requires approximately 9 GiB,
        in addition to other arrays and workspaces.
    Plotting from saved results: MATLAB; no GPU computation is required.


2. Directory Structure and Paper Correspondence

The repository is organized according to the four numerical experiments in Section 6 of the manuscript.

    run_experiment.m
        Main entry for computing the numerical data of Experiments 1-4.

    plot_experiment.m
        Main entry for generating figures from locally computed results.

    reproduce_figures.m
        Generates all four figure groups after their data have been computed.

    experiment_context.m
        Loads one experiment directory at a time for the main entries.

    exp1/
        All Experiment 1 MATLAB functions, Section 6.1, Figure 1(a-c).

    exp2/
        All Experiment 2 MATLAB functions, Section 6.2, Figure 2(a-b).

    exp3/
        All Experiment 3 MATLAB functions, Section 6.3, Figure 3(a-b).

    exp4/
        All Experiment 4 MATLAB functions, Section 6.4, Figure 4(a-c).

All MATLAB functions belonging to an experiment are located directly in its
exp1, exp2, exp3 or exp4 directory. No numerical data or figures are included.
Result, figure and checkpoint directories are created automatically during
execution inside the corresponding experiment directory.


3. Main MATLAB Functions and Figure Data

The function paths below are relative to the corresponding experiment directory. Run them through the root-level commands listed in Section 5.

Figure 1(a): Outer iterations of OSI, NRI and ERI along the Bloch path

    run_fig1a_A_only.m
        Computes the original shift-and-invert realization (OSI, Method A)
        at 12 Bloch points for N=5 and saves fig1a_A.mat.

    run_fig1a_BC_method.m
        Computes the nested reduced inverse realization (NRI, Method B)
        or explicit reduced inverse realization (ERI, Method C).
        Saves fig1a_B.mat or fig1a_C.mat.

    plot_fig1a_final.m
        Reads these three MAT files and plots the outer iteration counts
        for Figure 1(a).

    Data directory: exp1/results/

Figure 1(b) and Figure 1(c): Linear iteration counts and solver wall time

    run_fig1bc_method.m
        Computes Method A, B or C at the fixed Bloch vector
        q=(0.05,0.04,0.03) for N=3:7.
        Saves fig1bc_A.mat, fig1bc_B.mat or fig1bc_C.mat, including
        outer iteration counts, linear iteration counts and solver times.

    plot_fig1bc.m
        Reads these three MAT files.
        Generates Figure 1(b), showing average linear iterations per outer
        step, and Figure 1(c), showing solver wall time versus 3*NF.

    Data directory: exp1/results/

Figure 2(a): Accuracy-cost trade-off of multi-center Taylor reconstruction

    RUN_EXPERIMENT2_FIG2A.m
    run_experiment2_allgpu.m
        Compute the first ten positive eigenmodes at N=5 and use mode 10
        for reconstruction with h=0.0125 and n=40.
        Construct the blockwise direct reference and evaluate Taylor orders
        p=6,8,10 with m^3 centers for m=1,...,6.
        Save reconstruction errors and GPU timings in
        Experiment2_N5_n40_AllGPU_results.mat and
        Experiment2_N5_Fig2a_tradeoff_GPU.csv.

    plot_experiment2a_tradeoff.m
        Generates Figure 2(a) from the reconstruction errors and timings.

    Data directory: exp2/output/

Figure 2(b): Reconstruction scalability with increasing sampling window

    RUN_EXPERIMENT2_FIG2B.m
    run_experiment2b_ng_scaling.m
        Compare direct and Taylor reconstruction for n=40:20:200 at
        N=5 and h=0.0125, using Taylor order p=10.
        Save reconstruction timings, field errors and completion statuses in
        Experiment2b_N5_NgScaling_results.mat and
        Experiment2b_N5_NgScaling.csv.

    plot_experiment2b_ng_scaling.m
        Generates Figure 2(b), using n^3 as the horizontal coordinate.

    Data directory: exp2/output/

Figure 3(a): Physical-space consistency under mesh and window refinement

    RUN_EXP3.m
    run_experiment4.m
        Compute or load the selected N=5 modes 11 and 114 and reconstruct
        their three-dimensional Yee fields.
        Evaluate the relative Yee residual, LWRQ variation and relative
        3D-6D spectral discrepancy for the mesh and window studies.
        Save Experiment3_N5_AllGPU_results.mat,
        Experiment3_N5_mesh_sensitivity.csv and
        Experiment3_N5_window_sensitivity.csv.

    plot_experiment3a_physical_consistency.m
        Generates Figure 3(a) from these mesh and window results.

    Data directory: exp3/output/

Figure 3(b): Spatial distribution of local LWRQ deviations

    RUN_EXP3.m
    run_experiment4.m
        Also compute the relative local LWRQ deviations of modes 11 and 114
        on the z=0 and x=0 sections of the window [-4,4]^3.
        Save the spatial slices in Experiment3_N5_AllGPU_results.mat.

    plot_experiment3b_lwrq_slices.m
        Generates Figure 3(b) from the saved spatial slices.

    PLOT_EXP3.m generates both Figure 3(a) and Figure 3(b).

Figure 4(a): Low-frequency spectrum and LWRQ-recovered spectral values

    run_exp4_medium.m
        With the 'PAPER' setting, uses the permittivity in equation (6.2).
        Computes 20 positive eigenvalues at each of 80 independent Bloch
        points and reconstructs the four selected point-mode pairs.
        Saves Exp4_PAPER_results.mat, Exp4_PAPER_spectrum.csv,
        Exp4_PAPER_selected.csv and Exp4_PAPER_timing.csv.

    plot_exp4_spectrum_single.m
        Generates Figure 4(a), showing the computed spectrum and the
        LWRQ-recovered values at the four selected pairs.

    Data directory: exp4/output/

Figure 4(b) and Figure 4(c): Reconstructed electric-field intensities

    run_exp4_medium.m
        Also reconstructs the selected electric fields on enlarged z=0
        and x=y sections over [-2*pi,2*pi].
        Saves the field slices in Exp4_PAPER_results.mat.

    plot_exp4_field_slice_group.m
        Generates Figure 4(b) on z=0 and Figure 4(c) on x=y.

    PLOT_PAPER.m generates all three panels of Figure 4.


4. Description of Numerical Experiments

Experiment 1: Comparison of original and reduced inverse realizations

    This experiment compares OSI, NRI and ERI for the same projected
    Maxwell eigenvalue problem. Figure 1(a) compares their outer iteration
    counts along a closed Bloch path. Figures 1(b) and 1(c) examine the
    linear-solver work and wall time as the Fourier truncation increases.

    The scaling study uses a limit of 10^6 linear iterations or 10^4 solver
    seconds for each method and size. Larger sizes are skipped after a
    method reaches a stopping limit.

Experiment 2: Three-dimensional Yee-field reconstruction

    This experiment compares blockwise direct evaluation with multi-center
    Taylor reconstruction. Figure 2(a) examines the effects of Taylor order
    and the number of local centers. Figure 2(b) examines reconstruction
    cost and field error as the physical sampling window increases.

    The adopted setting is p=10 with 64 centers at n=40. For larger windows,
    the partition is selected to maintain the same phase-radius bound.
    Direct and Taylor reconstruction each use a 10^4-second time budget.

Experiment 3: Physical-space consistency and LWRQ validation

    This experiment validates the reconstructed fields against an
    independently discretized three-dimensional Yee operator. It compares
    the relative Yee residual, the variation of local Rayleigh quotients
    and the discrepancy between recovered 3D and projected 6D eigenvalues.

    Mesh refinement uses h=0.1,0.05,0.025,0.0125 on a fixed physical window.
    Window sensitivity is evaluated at h=0.025 with half-widths from 0.25
    to 4. The local quotient distributions are shown for modes 11 and 114.

Experiment 4: A representative quasiperiodic Maxwell problem

    This experiment applies the full method to the medium in equation
    (6.2). It computes the low-frequency spectrum along a closed Bloch
    path and selects the point-mode pairs (11,1), (31,9), (51,12) and
    (71,18) for field reconstruction and LWRQ recovery.

    Yee validation uses h=0.025 and n=20. Enlarged z=0 and x=y sections
    display the normalized reconstructed electric-field intensities.


5. How to Run

1. Open MATLAB and navigate to the root directory of this repository.

2. Add only the root directory to the MATLAB path:

       addpath(pwd);

   The main entry functions load the required experiment directories.
   Do not add all experiment subdirectories simultaneously, because they
   contain functions with the same names.

3. Compute the numerical data for the required figures:

   Figure 1(a):
       run_experiment(1,'1a-A');
       run_experiment(1,'1a-B');
       run_experiment(1,'1a-C');

   Figure 1(b,c):
       run_experiment(1,'1bc-A');
       run_experiment(1,'1bc-B');
       run_experiment(1,'1bc-C');

   Figure 2(a):
       run_experiment(2,'2a');

   Figure 2(b):
       run_experiment(2,'2b');

   Figure 3(a,b):
       run_experiment(3);

   Figure 4(a-c):
       run_experiment(4);

4. Generate figures from the computed results:

       plot_experiment(1);
       plot_experiment(2);
       plot_experiment(3);
       plot_experiment(4);

   Each command plots the full figure group for that experiment and
   requires the corresponding data files listed in Section 3.
   Figures are saved in exp1/figures/, exp2/figures/, exp3/figure/ and
   exp4/figure/, respectively.

   After computing all four experiments, the figures can also be generated
   together by running:

       reproduce_figures;

5. Numerical results are saved in exp1/results/, exp2/output/, exp3/output/
   and exp4/output/. MAT files contain the results required for plotting;
   CSV and text files contain numerical summaries and running times.
   No precomputed results are included, so the computation commands must
   be run before the plotting commands. Experiment 3 computes its selected
   eigenpairs automatically when no local checkpoint is available.


6. Citation

If you use this code, please cite the associated manuscript:

    Teng-Chao Sun, Tiexiang Li, Wen-Wei Lin, and Xing-Long Lyu.
    "Null-Space Free 6D Projection Method with Local Rayleigh Quotient
    Recovery for 3D Quasiperiodic Maxwell's Eigenproblems."
