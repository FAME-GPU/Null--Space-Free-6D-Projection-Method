function rho = baseline_phase_radius_exp2b(n,h,k,qL1Max)
%BASELINE_PHASE_RADIUS_EXP2B Exact radius for the adopted equal partition.
    m = 2*n + 3;
    Lmax = ceil(m/k);
    rho = 0.5*h*(Lmax-1)*qL1Max;
end
