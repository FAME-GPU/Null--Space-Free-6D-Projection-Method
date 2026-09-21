function [Usel,theta,iter,info,selIdx] = Lanczos_tracking( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose, ...
    fixedIndices,targetLambda,candidateHalfWidth)
%LANCZOS_TRACKING User Lanczos recurrence with memory-safe selected vectors.
%
% The Krylov recurrence, restart strategy, Gram-Schmidt, Ritz extraction,
% and stopping criterion are the same as the user's restarted Lanczos.
% The only tracking-specific change is at the final extraction: instead of
% forming all k Ritz vectors, only requested/candidate Ritz vectors are
% formed.  This avoids a multi-gigabyte k-vector output for the N=5 scan.

    % The function has 14 input arguments.  Keep the optional-argument
    % guards aligned with the actual argument positions.
    if nargin<11 || isempty(verbose), verbose=false; end
    if nargin<12, fixedIndices=[]; end
    if nargin<13, targetLambda=[]; end
    if nargin<14 || isempty(candidateHalfWidth), candidateHalfWidth=0; end

    if flag_gpu && ~isa(v0,'gpuArray'), v0=gpuArray(v0); end
    nrm0=sqrt(abs(gather_scalar(v0'*fncB(v0))));
    v0=v0/nrm0;

    [theta,Usel,iter,info,selIdx]=Lanczos_iteration_tracking( ...
        fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose, ...
        fixedIndices,targetLambda,candidateHalfWidth);
end

function [d,Usel,kk,info,selIdx] = Lanczos_iteration_tracking( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v,flag_gpu,verbose, ...
    fixedIndices,targetLambda,candidateHalfWidth)

    eps0=eps('double');
    Q=zeros(n,p,'like',v);
    k0=k;

    Alpha=zeros(0,1); Beta=zeros(0,1); beta=0;
    d=zeros(0,1); c=zeros(0,1);
    justRestarted=false; sizeV=1; kk=0;

    info.restart=[]; info.totalIter=[];
    info.maxWantedResidual=[]; info.meanWantedResidual=[];
    info.converged=false; info.numConverged=0;
    info.hitMaxRestart=false; info.finalRestart=0;

    Usmall=[]; ind=[]; res=[];
    runTimer=tic;

    for mm=1:maxiter
        for jj=sizeV:p
            tIter=tic;
            Q(:,jj)=v;
            Aq=fncA(Q(:,jj));
            alpha=real(gather_scalar(Aq'*Q(:,jj)));
            pr=fncBinv(Aq);

            if jj==1
                r=pr-alpha*Q(:,jj);
            elseif justRestarted
                r=GramSchmitGEP_tracking(Q,pr,fncB,jj);
                justRestarted=false;
            else
                r=pr-alpha*Q(:,jj)-beta*Q(:,jj-1);
                corr=gather_scalar(fncB(Q(:,jj-1))'*r);
                r=r-corr*Q(:,jj-1);
            end

            corr=gather_scalar(fncB(Q(:,jj))'*r);
            r=r-corr*Q(:,jj);
            r=GramSchmitGEP_tracking(Q,r,fncB,jj);
            r=GramSchmitGEP_tracking(Q,r,fncB,jj);

            beta2=real(gather_scalar(r'*fncB(r)));
            if beta2<=0 || sqrt(beta2)<1e-14
                warning('Lanczos breakdown or near breakdown at step %d.',kk);
                beta=sqrt(max(beta2,0));
                break;
            end

            beta=sqrt(beta2);
            v=r/beta;
            Alpha(end+1,1)=alpha; %#ok<AGROW>
            Beta(end+1,1)=beta; %#ok<AGROW>
            kk=kk+1;
            if verbose
                wait_for_gpu();
                fprintf('Lanczos tracking iter %d, %.2f s.\n',kk,toc(tIter));
            end
        end

        H1=diag(Alpha)+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
        H=blkdiag(diag(d),H1);
        if ~isempty(d)
            H(1:k,k+1)=c'; H(k+1,1:k)=c;
        end

        Alpha=zeros(0,1); Beta=zeros(0,1);
        [Usmall,dRaw]=eig(H,'vector');
        resRaw=abs(beta*Usmall(end,:));
        [~,ind]=sort(real(dRaw),'descend');
        d=dRaw(ind); res=resRaw(ind);

        wanted=1:min(k0,numel(res));
        info.restart(end+1)=mm;
        info.totalIter(end+1)=kk;
        info.maxWantedResidual(end+1)=max(res(wanted));
        info.meanWantedResidual(end+1)=mean(res(wanted));

        wantedRes=res(wanted).'; wantedD=d(wanted);
        threshold=tol*max(eps0^(2/3),abs(wantedD));
        nconv=nnz(wantedRes<threshold);
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

        keep=ind(1:k);
        Ukeep=Usmall(:,keep);
        if flag_gpu
            Q(:,1:k)=Q*gpuArray(cast(Ukeep,classUnderlying(Q)));
        else
            Q(:,1:k)=Q*Ukeep;
        end
        d=d(1:k);
        c=beta*Ukeep(end,:);
        justRestarted=true;
        sizeV=k+1;
    end

    d=d(1:k0);
    res=res(1:k0);
    lambda=real(1./d(:));

    if ~isempty(fixedIndices)
        selIdx=unique(fixedIndices(:).','stable');
        if any(selIdx<1) || any(selIdx>k0)
            error('Requested fixed tracking index is outside the scan.');
        end
    else
        if isempty(targetLambda)
            error('Either fixedIndices or targetLambda must be supplied.');
        end
        selIdx=[];
        for t=targetLambda(:).'
            [~,ic]=min(abs(lambda-t)./max(abs(t),eps));
            lo=max(1,ic-candidateHalfWidth);
            hi=min(k0,ic+candidateHalfWidth);
            selIdx=union(selIdx,lo:hi,'stable');
        end
        if numel(selIdx)<2
            error(['Tracking candidate extraction produced only %d candidate(s). ', ...
                'candidateHalfWidth=%d.'],numel(selIdx),candidateHalfWidth);
        end
        fprintf('  tracking candidate extraction: halfWidth=%d, candidates=%d, index range=[%d,%d]\n', ...
            candidateHalfWidth,numel(selIdx),min(selIdx),max(selIdx));
    end

    % Form only the selected Ritz vectors.  This is the crucial memory-safe
    % difference from the standard Lanczos return when k is large.
    Us=Usmall(:,ind(selIdx));
    if flag_gpu
        Usel=Q*gpuArray(cast(Us,classUnderlying(Q)));
    else
        Usel=Q*Us;
    end

    info.wallTimeSeconds=toc(runTimer);
    info.allRitzResiduals=res(:);
    info.allRelativeInverseRitzResiduals=res(:)./max(abs(d(:)),eps);
    info.selectedIndices=selIdx(:);
    info.selectedRelativeInverseRitzResiduals= ...
        info.allRelativeInverseRitzResiduals(selIdx);
    info.lambda=lambda;
end

function r=GramSchmitGEP_tracking(V,r,fncB,jj)
    for kk=1:jj
        tmp=gather_scalar(fncB(V(:,kk))'*r);
        r=r-tmp*V(:,kk);
    end
end
