function [Usel,theta,iter,info,selIdx] = Lanczos_spectrum_selected( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose,selectedIndices)
%LANCZOS_SPECTRUM_SELECTED Restarted Lanczos, GPU production path.
%
% Large vectors, orthogonalization inner products, recurrence updates, and
% the projected dense eigensolve are evaluated on GPU. Only small scalars
% and the tiny projected matrices/eigenvectors are gathered for restart
% control and diagnostics.

    if nargin<11 || isempty(verbose), verbose=false; end
    if nargin<12, selectedIndices=[]; end
    if ~flag_gpu || ~isa(v0,'gpuArray')
        error('Experiment 4 production Lanczos requires a gpuArray start vector.');
    end

    nrmGPU=sqrt(abs(v0'*fncB(v0)));
    nrm0=gather_scalar(nrmGPU);
    if ~isfinite(nrm0) || nrm0==0, error('Invalid initial Lanczos vector.'); end
    v0=v0./nrmGPU;

    [theta,Usel,iter,info,selIdx]=iteration( ...
        fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,verbose,selectedIndices);
end

function [dGPU,Usel,kk,info,selIdx] = iteration( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v,verbose,selectedIndices)

    eps0=eps('double');
    Q=complex(zeros(n,p,'like',v));
    k0=k;
    Alpha=zeros(0,1); Beta=zeros(0,1);
    betaGPU=cast(0,'like',real(v(1)));
    dCPU=zeros(0,1); cCPU=zeros(0,1);
    justRestarted=false; sizeV=1; kk=0;

    info.restart=[]; info.totalIter=[];
    info.maxWantedResidual=[]; info.meanWantedResidual=[];
    info.converged=false; info.numConverged=0;
    info.hitMaxRestart=false; info.finalRestart=0;
    resCPU=[]; UsmallCPU=[]; indCPU=[];
    runTimer=tic;

    for mm=1:maxiter
        for jj=sizeV:p
            tIter=tic;
            Q(:,jj)=v;
            Aq=fncA(Q(:,jj));
            alphaGPU=real(Aq'*Q(:,jj));
            alphaCPU=gather_scalar(alphaGPU);
            pr=fncBinv(Aq);

            if jj==1
                r=pr-alphaGPU.*Q(:,jj);
            elseif justRestarted
                r=gs_gpu(Q,pr,fncB,jj);
                justRestarted=false;
            else
                r=pr-alphaGPU.*Q(:,jj)-betaGPU.*Q(:,jj-1);
                corrGPU=fncB(Q(:,jj-1))'*r;
                r=r-corrGPU.*Q(:,jj-1);
            end

            corrGPU=fncB(Q(:,jj))'*r;
            r=r-corrGPU.*Q(:,jj);
            r=gs_gpu(Q,r,fncB,jj);
            r=gs_gpu(Q,r,fncB,jj);

            beta2GPU=real(r'*fncB(r));
            beta2CPU=gather_scalar(beta2GPU);
            if beta2CPU<=0 || sqrt(beta2CPU)<1e-14
                warning('Lanczos breakdown or near breakdown at step %d.',kk);
                betaGPU=sqrt(max(beta2GPU,cast(0,'like',beta2GPU)));
                break;
            end

            betaGPU=sqrt(beta2GPU);
            betaCPU=gather_scalar(betaGPU);
            v=r./betaGPU;
            Alpha(end+1,1)=alphaCPU; %#ok<AGROW>
            Beta(end+1,1)=betaCPU; %#ok<AGROW>
            kk=kk+1;
            if verbose
                wait_for_gpu();
                fprintf('Lanczos Exp4 GPU iter %d, %.2f s.\n',kk,toc(tIter));
            end
        end

        H1=diag(Alpha)+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
        H=blkdiag(diag(dCPU),H1);
        if ~isempty(dCPU)
            H(1:k,k+1)=cCPU'; H(k+1,1:k)=cCPU;
        end
        Alpha=zeros(0,1); Beta=zeros(0,1);

        % Dense projected eigensolve is also executed on GPU.
        Hgpu=gpuArray(cast(H,classUnderlying(Q)));
        [UsmallGPU,dRawGPU]=eig(Hgpu,'vector');
        resRawGPU=abs(betaGPU.*UsmallGPU(end,:));
        [~,indGPU]=sort(real(dRawGPU),'descend');
        indCPU=gather(indGPU);
        dSortedGPU=dRawGPU(indCPU);
        resSortedGPU=resRawGPU(indCPU);
        UsortedGPU=UsmallGPU(:,indCPU);
        dCPU=gather(dSortedGPU);
        resCPU=gather(resSortedGPU).';
        UsmallCPU=gather(UsortedGPU);
        clear Hgpu UsmallGPU dRawGPU resRawGPU indGPU dSortedGPU resSortedGPU UsortedGPU H H1

        wanted=1:min(k0,numel(resCPU));
        info.restart(end+1)=mm;
        info.totalIter(end+1)=kk;
        info.maxWantedResidual(end+1)=max(resCPU(wanted));
        info.meanWantedResidual(end+1)=mean(resCPU(wanted));

        nconv=count_converged_ritz(resCPU(wanted),dCPU(wanted),tol);
        info.numConverged=nconv; info.finalRestart=mm;

        if nconv>=k0
            info.converged=true;
            break;
        elseif mm==maxiter
            info.hitMaxRestart=true;
            break;
        end

        k=k0+min(nconv,floor((p-k0)/2));
        if k==1 && p>3, k=floor(p/2); end
        UkeepCPU=UsmallCPU(:,1:k);
        Q(:,1:k)=Q*gpuArray(cast(UkeepCPU,classUnderlying(Q)));
        dCPU=dCPU(1:k);
        cCPU=gather_scalar(betaGPU).*UkeepCPU(end,:);
        justRestarted=true;
        sizeV=k+1;
    end

    if numel(dCPU)<k0
        error('Lanczos returned only %d Ritz values; %d required.',numel(dCPU),k0);
    end
    dCPU=dCPU(1:k0); resCPU=resCPU(1:k0);
    selIdx=unique(selectedIndices(:).','stable');
    if any(selIdx<1) || any(selIdx>k0)
        error('Requested selected Ritz index is outside 1:k.');
    end

    if isempty(selIdx)
        Usel=complex(zeros(n,0,'like',v));
    else
        UsCPU=UsmallCPU(:,selIdx);
        Usel=Q*gpuArray(cast(UsCPU,classUnderlying(Q)));
    end

    dGPU=gpuArray(cast(dCPU,classUnderlying(Q)));
    info.wallTimeSeconds=toc(runTimer);
    info.allRitzResiduals=resCPU(:);
    info.allRelativeInverseRitzResiduals=resCPU(:)./max(abs(dCPU(:)),eps);
    info.selectedIndices=selIdx(:);
    info.selectedRelativeInverseRitzResiduals= ...
        info.allRelativeInverseRitzResiduals(selIdx);
end

function r=gs_gpu(V,r,fncB,jj)
    for kk=1:jj
        tmpGPU=fncB(V(:,kk))'*r;
        r=r-tmpGPU.*V(:,kk);
    end
end
