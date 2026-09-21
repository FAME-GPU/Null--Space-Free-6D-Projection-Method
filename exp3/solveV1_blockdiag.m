function y = solveV1_blockdiag( ...
    x,V1_11_diag,V1_12_diag,V1_21_diag,V1_22_diag,V1_det_diag,n)
%SOLVEV1_BLOCKDIAG GPU modewise solve y=V1^{-1}x.

    x1 = x(1:n,:);
    x2 = x(n+1:2*n,:);
    y1 = (V1_22_diag.*x1 - V1_12_diag.*x2)./V1_det_diag;
    y2 = (-V1_21_diag.*x1 + V1_11_diag.*x2)./V1_det_diag;
    y = [y1;y2];
end
