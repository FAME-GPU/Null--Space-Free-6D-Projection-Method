function y = apply_V_blockdiag(x,Vdata)
%APPLY_V_BLOCKDIAG Apply V=Ur*Sigma^(1/2) from six modewise diagonals.

    n = Vdata.n;
    if size(x,1) ~= 2*n
        error('apply_V_blockdiag input must have 2*n rows.');
    end

    x1 = x(1:n,:);
    x2 = x(n+1:2*n,:);

    y1 = Vdata.V11.*x1 + Vdata.V12.*x2;
    y2 = Vdata.V21.*x1 + Vdata.V22.*x2;
    y3 = Vdata.V31.*x1 + Vdata.V32.*x2;

    y = [y1;y2;y3];
end
