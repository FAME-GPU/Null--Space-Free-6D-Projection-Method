function [Xgpu,info] = recover_fixed_vectors_tracking(Y,sys,cfg)
%RECOVER_FIXED_VECTORS_TRACKING Recover selected Fourier eigenvectors on GPU.
    if ~isa(Y,'gpuArray')
        error('Selected-vector recovery requires GPU Lanczos vectors.');
    end
    k=size(Y,2);
    Xgpu=complex(zeros(sys.ndof,k,'like',Y));
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
        n2=real(xj'*Mx);
        n2CPU=gather_scalar(n2);
        if ~isfinite(n2CPU) || n2CPU<=0, error('Bad M norm in tracking recovery.'); end
        xj=xj./sqrt(n2);
        Xgpu(:,j)=xj;
        iterations(j)=itCPU; relres(j)=rrCPU;
        clear rhs x0 xj Mx n2
    end

    info=struct('iterations',iterations,'relativeResiduals',relres,'device','GPU');
end
