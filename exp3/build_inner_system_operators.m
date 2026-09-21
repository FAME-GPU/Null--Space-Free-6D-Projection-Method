function ops = build_inner_system_operators( ...
    mass,Wop,WHop,W1,W2, ...
    V11,V12,V21,V22,detV1,n)
%BUILD_INNER_SYSTEM_OPERATORS Build S, Mhat, B_r and B_r^* actions.
%
% H = I + W^*W = C^*C is diagonal mode by mode in the present basis.
% Mhat = C^{-*} S C^{-1}.
% B_r = V1^{-1}(M3-M1 W) C^{-1}.

    KorthDiag = real(1 + abs(W1).^2 + abs(W2).^2);
    if gather_scalar(min(KorthDiag)) <= 0
        error('Korth has a nonpositive diagonal entry.');
    end
    invCorthDiag = 1./sqrt(KorthDiag);

    ops = struct();
    ops.KorthDiag = KorthDiag;
    ops.invCorthDiag = invCorthDiag;
    ops.Sop = @applyS;
    ops.MhatOp = @applyMhat;
    ops.BrOp = @applyBr;
    ops.BrHop = @applyBrH;

    function y = applyS(x)
        Wx = Wop(x);

        Mcol3x = mass.Mcol3op(x);
        M3x = Mcol3x(1:2*n,:);
        M2x = Mcol3x(2*n+1:3*n,:);

        Mcols12Wx = mass.Mcols12op(Wx);
        M1Wx = Mcols12Wx(1:2*n,:);
        M3HWx = Mcols12Wx(2*n+1:3*n,:);

        y = M2x - M3HWx - WHop(M3x) + WHop(M1Wx);
    end

    function y = applyMhat(x)
        z = invCorthDiag.*x;
        y = invCorthDiag.*applyS(z);
    end

    function y = applyBr(x)
        z = invCorthDiag.*x;

        Mcol3z = mass.Mcol3op(z);
        M3z = Mcol3z(1:2*n,:);

        Wz = Wop(z);
        Mcols12Wz = mass.Mcols12op(Wz);
        M1Wz = Mcols12Wz(1:2*n,:);

        d = M3z-M1Wz;
        y = solveV1_blockdiag( ...
            d,V11,V12,V21,V22,detV1,n);
    end

    function y = applyBrH(q)
        r = solveV1H_blockdiag( ...
            q,V11,V12,V21,V22,detV1,n);

        Mcols12r = mass.Mcols12op(r);
        M1r = Mcols12r(1:2*n,:);
        M3Hr = Mcols12r(2*n+1:3*n,:);

        b = M3Hr-WHop(M1r);
        y = invCorthDiag.*b;
    end
end
