function [theta,iter,info] = Lanczos_hybrid_identity_gpuop( ...
    fncA_gpu,n,k,tol,maxRestart,p,v0CPU,precision,verbose,budget)
%LANCZOS_HYBRID_IDENTITY_GPUOP Restarted Lanczos with CPU outer basis.
%
% The reduced eigenproblem uses the Euclidean inner product.  Krylov basis,
% reorthogonalization, projected eigensolves and restart algebra are kept on
% CPU. Each inverse action is transferred to GPU, evaluated by FNCA_GPU, and
% gathered back. The mathematical recurrence/restart policy matches the
% previous identity-mass Lanczos implementation.

    if nargin<9 || isempty(verbose), verbose=false; end
    if nargin<10 || isempty(budget), budget=make_run_budget(Inf,Inf); end
    if isa(v0CPU,'gpuArray'), v0CPU=gather(v0CPU); end
    v=cast(v0CPU(:),precision);
    nv=norm(v,2); if ~(isfinite(nv)&&nv>0), error('Invalid Lanczos start vector.'); end
    v=v/nv;

    eps0=eps('double');
    qPrototype=cast(1i,precision);
    Q=zeros(n,p,'like',qPrototype);
    k0=k; Alpha=zeros(0,1); Beta=zeros(0,1); beta=0;
    d=zeros(0,1); c=zeros(0,1); justRestarted=false; sizeV=1; kk=0;
    Usmall=[]; ind=[]; res=[];

    info=struct('restart',[],'totalIter',[],'maxWantedResidual',[], ...
        'meanWantedResidual',[],'ritzValues',{{}},'ritzResiduals',{{}}, ...
        'converged',false,'numConverged',0,'hitMaxRestart',false, ...
        'finalRestart',0,'totalH2DSeconds',0,'totalD2HSeconds',0, ...
        'totalGPUOperatorSeconds',0,'wallTimeSeconds',0);

    runTimer=tic;
    for mm=1:maxRestart
        for jj=sizeV:p
            budget.checkCanContinue('hybrid Lanczos outer step');
            Q(:,jj)=v;

            h2d=tic; qg=to_gpu(Q(:,jj),precision); wait_for_gpu(qg);
            info.totalH2DSeconds=info.totalH2DSeconds+toc(h2d);

            opTimer=tic; Aqg=fncA_gpu(qg); wait_for_gpu(Aqg);
            info.totalGPUOperatorSeconds=info.totalGPUOperatorSeconds+toc(opTimer);

            d2h=tic; Aq=gather(Aqg); info.totalD2HSeconds=info.totalD2HSeconds+toc(d2h);
            clear qg Aqg

            if size(Aq,1)~=n || size(Aq,2)~=1 || any(~isfinite(Aq))
                error('Hybrid Lanczos inverse action returned invalid CPU data.');
            end
            alpha=real(Aq'*Q(:,jj));
            pr=Aq;

            if jj==1
                r=pr-alpha*Q(:,jj);
            elseif justRestarted
                r=gram_schmidt_cpu(Q,pr,jj);
                justRestarted=false;
            else
                r=pr-alpha*Q(:,jj)-beta*Q(:,jj-1);
                corr=Q(:,jj-1)'*r;
                r=r-corr*Q(:,jj-1);
            end
            corr=Q(:,jj)'*r; r=r-corr*Q(:,jj);
            r=gram_schmidt_cpu(Q,r,jj);
            r=gram_schmidt_cpu(Q,r,jj);

            beta2=real(r'*r);
            if beta2<=0 || sqrt(beta2)<1e-14
                warning('Lanczos breakdown or near breakdown at step %d.',kk);
                beta=sqrt(max(beta2,0));
                break
            end
            beta=sqrt(beta2); v=r/beta;
            Alpha(end+1,1)=alpha; %#ok<AGROW>
            Beta(end+1,1)=beta; %#ok<AGROW>
            kk=kk+1;
            if verbose, fprintf('Hybrid Lanczos iter %d\n',kk); end
        end

        if isempty(Alpha) && isempty(d)
            error('Lanczos produced no projected data.');
        end
        H1=diag(Alpha);
        if numel(Beta)>1
            H1=H1+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
        end
        H=blkdiag(diag(d),H1);
        if ~isempty(d)
            H(1:k,k+1)=c'; H(k+1,1:k)=c;
        end
        Alpha=zeros(0,1); Beta=zeros(0,1);
        [Usmall,d]=eig(H,'vector');
        res=abs(beta*Usmall(end,:));
        [~,ind]=sort(real(d),'descend'); d=d(ind); res=res(ind);

        wanted=1:min(k0,numel(res));
        if isempty(wanted), error('Lanczos projected problem has no wanted Ritz values.'); end
        info.restart(end+1)=mm; %#ok<AGROW>
        info.totalIter(end+1)=kk; %#ok<AGROW>
        info.maxWantedResidual(end+1)=max(res(wanted)); %#ok<AGROW>
        info.meanWantedResidual(end+1)=mean(res(wanted)); %#ok<AGROW>
        info.ritzValues{end+1}=d(wanted); %#ok<AGROW>
        info.ritzResiduals{end+1}=res(wanted); %#ok<AGROW>

        wantedRes=res(wanted).'; wantedD=d(wanted);
        threshold=tol*max(eps0^(2/3),abs(wantedD));
        nconv=nnz(wantedRes<threshold);
        info.numConverged=nconv; info.finalRestart=mm;
        if nconv>=k0
            info.converged=true; break
        elseif mm==maxRestart
            info.hitMaxRestart=true; break
        end

        k=k0+min(nconv,floor((p-k0)/2));
        if k==1 && p>3, k=floor(p/2); end
        keep=ind(1:k); Ukeep=Usmall(:,keep);
        qcols=size(Ukeep,1);
        Q(:,1:k)=Q(:,1:qcols)*Ukeep;
        d=d(1:k); c=beta*Ukeep(end,:);
        justRestarted=true; sizeV=k+1;
    end

    if isempty(ind) || numel(ind)<k0
        error('Lanczos ended without %d inverse Ritz values.',k0);
    end
    theta=real(d(1:k0));
    iter=kk;
    info.wallTimeSeconds=toc(runTimer);
    clear Q
end

function r=gram_schmidt_cpu(V,r,jj)
    for ii=1:jj
        r=r-(V(:,ii)'*r)*V(:,ii);
    end
end
