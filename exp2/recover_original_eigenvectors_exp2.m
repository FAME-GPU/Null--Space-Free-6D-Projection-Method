function [X,MX,info] = recover_original_eigenvectors_exp2(Y,sys,cfg)
%RECOVER_ORIGINAL_EIGENVECTORS_EXP2 Map reduced Ritz vectors to original GEVP.
%
% Memory-safe N=5 implementation:
%   Each recovered eigenvector is normalized column-by-column.  The mass
%   action M*x_j needed for the M-norm is reused to form the corresponding
%   normalized column of MX.  This deliberately avoids a final multi-RHS
%   call MX = sys.mass.Mop(X), which would embed all eigenvectors and all
%   three field components simultaneously on the rank-1 lattice and can
%   exceed the memory of a 32-GB V100 at N=5.

    k=size(Y,2);
    X=complex(zeros(sys.ndof,k,'like',Y));
    MX=complex(zeros(sys.ndof,k,'like',Y));
    recoveryIterations=zeros(k,1);
    recoveryRelres=zeros(k,1);

    for j=1:k
        rhs=apply_V_blockdiag(Y(:,j),sys.Vdata);
        x0=zeros(size(rhs),'like',rhs);
        [xj,flag,rr,it]=pcg(@(x)sys.mass.Mop(x),rhs, ...
            cfg.solver.tolRecoverM,cfg.solver.maxitRecoverM,[],[],x0);

        flagCPU=gather_scalar(flag);
        rrCPU=gather_scalar(rr);
        finiteX=gather_scalar(all(isfinite(xj(:))));
        if ~finiteX || ~isfinite(rrCPU) || ...
                (flagCPU~=0 && rrCPU>10*cfg.solver.tolRecoverM)
            error('recover_original_eigenvectors_exp2:RecoveryCGFailure', ...
                ['Recovery CG failed for eigenvector %d: flag=%d, ', ...
                 'relres=%.3e, finite=%d.'],j,flagCPU,rrCPU,finiteX);
        end

        % This mass action is already required for the M-normalization.
        Mxj=sys.mass.Mop(xj);
        nj2=real(gather_scalar(xj'*Mxj));
        if ~isfinite(nj2) || nj2<=0
            error('recover_original_eigenvectors_exp2:BadMassNorm', ...
                'Nonpositive/nonfinite M-norm for eigenvector %d.',j);
        end

        scale=sqrt(nj2);
        X(:,j)=xj/scale;
        MX(:,j)=Mxj/scale;

        recoveryIterations(j)=gather_scalar(it);
        recoveryRelres(j)=rrCPU;

        % Release the large temporary vectors before the next eigenvector.
        clear rhs x0 xj Mxj
    end

    info=struct();
    info.iterations=recoveryIterations;
    info.relativeResiduals=recoveryRelres;
    info.memorySafeColumnwiseMassAction=true;
end
