function qModes = build_projected_q_modes(P,k_lift,Xi,n,d)
%BUILD_PROJECTED_Q_MODES Build n-by-3 projected wave vectors on GPU.

    qModes = zeros(n,3,'like',Xi{1});
    for ell = 1:d
        xi_ell = Xi{ell}(:)+k_lift(ell);
        qModes(:,1) = qModes(:,1)+P(ell,1).*xi_ell;
        qModes(:,2) = qModes(:,2)+P(ell,2).*xi_ell;
        qModes(:,3) = qModes(:,3)+P(ell,3).*xi_ell;
    end
end
