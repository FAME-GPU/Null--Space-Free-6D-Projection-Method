function y = apply_VH_blockdiag(x,Vdata)
%APPLY_VH_BLOCKDIAG Apply V^* for V=Ur*Sigma^(1/2).

    n = Vdata.n;
    if size(x,1) ~= 3*n
        error('apply_VH_blockdiag input must have 3*n rows.');
    end

    x1 = x(1:n,:);
    x2 = x(n+1:2*n,:);
    x3 = x(2*n+1:3*n,:);

    y1 = conj(Vdata.V11).*x1 + conj(Vdata.V21).*x2 + conj(Vdata.V31).*x3;
    y2 = conj(Vdata.V12).*x1 + conj(Vdata.V22).*x2 + conj(Vdata.V32).*x3;
    y = [y1;y2];
end
