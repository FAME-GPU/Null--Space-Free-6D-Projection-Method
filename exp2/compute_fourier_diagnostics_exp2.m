function out = compute_fourier_diagnostics_exp2(X,MX,lambda,sys)
%COMPUTE_FOURIER_DIAGNOSTICS_EXP2 Original-GEVP residual and divergence.

    k=size(X,2);
    lambda=lambda(:);
    res=zeros(k,1);
    div=zeros(k,1);
    rayleigh=zeros(k,1);

    for j=1:k
        x=X(:,j); Mx=MX(:,j); Kx=sys.Kop(x);
        rayleigh(j)=gather_scalar(real((x'*Kx)/(x'*Mx)));
        denom=gather_scalar(norm(Kx,2))+abs(lambda(j))*gather_scalar(norm(Mx,2));
        res(j)=gather_scalar(norm(Kx-lambda(j)*Mx,2))/max(denom,eps);

        normalizedDiv=sys.U0Hop(Mx);
        GHMU=sys.D0vec.*normalizedDiv;
        div(j)=gather_scalar(norm(GHMU,2))/max(gather_scalar(norm(Mx,2)),eps);
    end

    out=struct();
    out.resGEVP=res;
    out.divRelative2=div;
    out.rayleigh=rayleigh;
    out.maxResGEVP=max(res);
    out.maxDivRelative2=max(div);
end
