function [Xcpu,info] = recover_fixed_vectors_tracking(Y,sys,cfg)
%RECOVER_FIXED_VECTORS_TRACKING Recover a small fixed set of eigenvectors.
% Returned vectors are stored on CPU to release GPU memory before N=5.

    k=size(Y,2);
    Xcpu=complex(zeros(sys.ndof,k,'double'));
    iterations=zeros(k,1); relres=zeros(k,1);

    for j=1:k
        rhs=apply_V_blockdiag(Y(:,j),sys.Vdata);
        x0=zeros(size(rhs),'like',rhs);
        [xj,flag,rr,it]=pcg(@(x)sys.mass.Mop(x),rhs, ...
            cfg.solver.tolRecoverM,cfg.solver.maxitRecoverM,[],[],x0);

        flagCPU=gather_scalar(flag); rrCPU=gather_scalar(rr); itCPU=gather_scalar(it);
        if ~gather_scalar(all(isfinite(xj(:)))) || ~isfinite(rrCPU) || ...
                (flagCPU~=0 && rrCPU>10*cfg.solver.tolRecoverM)
            error('Tracking recovery failed: j=%d, flag=%d, relres=%.3e.', ...
                j,flagCPU,rrCPU);
        end

        Mx=sys.mass.Mop(xj);
        n2=real(gather_scalar(xj'*Mx));
        if ~isfinite(n2) || n2<=0, error('Bad M norm in tracking recovery.'); end
        xj=xj/sqrt(n2);
        Xcpu(:,j)=gather(xj);
        iterations(j)=itCPU; relres(j)=rrCPU;
        clear rhs x0 xj Mx
    end

    info=struct('iterations',iterations,'relativeResiduals',relres);
end
