function y = apply_Mhat_C_scalar_compact(x,mass,W1,W2,invC)
%APPLY_MHAT_C_SCALAR_COMPACT Scalar-isotropic orthogonalized inner action.
%
% S x = Mx + W^* M W x because the physical mass matrix is
% diag(M,M,M) and M3=0.  This is algebraically identical to the generic
% block formula but avoids padded 3-component work arrays.

    z=invC.*x;
    s=mass.MscalarOp(z);

    t=W1.*z;
    t=mass.MscalarOp(t);
    s=s+conj(W1).*t;
    clear t

    t=W2.*z;
    t=mass.MscalarOp(t);
    s=s+conj(W2).*t;
    clear t z

    y=invC.*s;
end
