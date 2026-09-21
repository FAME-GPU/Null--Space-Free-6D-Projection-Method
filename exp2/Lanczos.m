function [ev,ew,iter,info] = Lanczos(fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose)
%LANCZOS Restarted GPU Krylov-Schur/Lanczos solver for a Hermitian GEP.
%
% This is the user's current Lanczos algorithm.  The Krylov recurrence,
% GEP Gram-Schmidt orthogonalization, restart, Ritz extraction, and stopping
% criterion are unchanged.  The only interface adaptation is the optional
% VERBOSE switch, which suppresses per-step fprintf during timing runs.

    if nargin<11 || isempty(verbose)
        verbose = true;
    end

    if flag_gpu && ~isa(v0,'gpuArray')
        v0 = gpuArray(v0);
    end

    nrm0 = sqrt(abs(gather_scalar(v0'*fncB(v0))));
    v0 = v0/nrm0;
    [ew,ev,iter,~,info] = Lanczos_iteration( ...
        fncA,fncB,fncBinv,n,k,tol,maxiter,p,v0,flag_gpu,verbose);
end

function [d,U,kk,res,info] = Lanczos_iteration( ...
    fncA,fncB,fncBinv,n,k,tol,maxiter,p,v,flag_gpu,verbose)

    eps0 = eps('double');
    Q = zeros(n,p,'like',v);
    k0 = k;

    Alpha = zeros(0,1);
    Beta = zeros(0,1);
    beta = 0;
    d = zeros(0,1);
    c = zeros(0,1);
    justRestarted = false;
    sizeV = 1;
    kk = 0;

    info.restart = [];
    info.totalIter = [];
    info.maxWantedResidual = [];
    info.meanWantedResidual = [];
    info.ritzValues = {};
    info.ritzResiduals = {};

    Usmall = [];
    ind = [];
    res = [];
    Num_iter = 0;
    for mm = 1:maxiter
        for jj = sizeV:p
            time1 = tic;
            Q(:,jj) = v;
            Aq = fncA(Q(:,jj));
            alpha = real(gather_scalar(Aq'*Q(:,jj)));
            pr = fncBinv(Aq);

            if jj==1
                r = pr-alpha*Q(:,jj);
            elseif justRestarted
                r = GramSchmitGEP(Q,pr,fncB,jj);
                justRestarted = false;
            else
                r = pr-alpha*Q(:,jj)-beta*Q(:,jj-1);
                corr = gather_scalar(fncB(Q(:,jj-1))'*r);
                r = r-corr*Q(:,jj-1);
            end

            corr = gather_scalar(fncB(Q(:,jj))'*r);
            r = r-corr*Q(:,jj);
            r = GramSchmitGEP(Q,r,fncB,jj);
            r = GramSchmitGEP(Q,r,fncB,jj);

            beta2 = real(gather_scalar(r'*fncB(r)));
            if beta2<=0 || sqrt(beta2)<1e-14
                warning('Lanczos breakdown or near breakdown at step %d.',kk);
                beta = sqrt(max(beta2,0));
                break;
            end

            beta = sqrt(beta2);
            v = r/beta;
            Alpha(end+1,1) = alpha; %#ok<AGROW>
            Beta(end+1,1) = beta; %#ok<AGROW>
            kk = kk+1;
            Num_iter = Num_iter + 1;
            if verbose
                fprintf('Lanczos iter num %d, spend time %.2f. \n', Num_iter, toc(time1));
            end
        end

        H1 = diag(Alpha)+diag(Beta(1:end-1),1)+diag(Beta(1:end-1),-1);
        H = blkdiag(diag(d),H1);
        if ~isempty(d)
            H(1:k,k+1) = c';
            H(k+1,1:k) = c;
        end

        Alpha = zeros(0,1);
        Beta = zeros(0,1);
        [Usmall,d] = eig(H,'vector');
        res = abs(beta*Usmall(end,:));
        [~,ind] = sort(real(d),'descend');
        d = d(ind);
        res = res(ind);

        wanted = 1:min(k0,numel(res));
        info.restart(end+1) = mm;
        info.totalIter(end+1) = kk;
        info.maxWantedResidual(end+1) = max(res(wanted));
        info.meanWantedResidual(end+1) = mean(res(wanted));
        info.ritzValues{end+1} = d(wanted);
        info.ritzResiduals{end+1} = res(wanted);

        wantedRes = res(wanted).';
        wantedD = d(wanted);
        threshold = tol*max(eps0^(2/3),abs(wantedD));
        nconv = nnz(wantedRes<threshold);
        if nconv>=k0 || mm==maxiter
            break;
        end

        k = k0+min(nconv,floor((p-k0)/2));
        if k==1 && p>3
            k = floor(p/2);
        end

        keep = ind(1:k);
        Ukeep = Usmall(:,keep);
        if flag_gpu
            Q(:,1:k) = Q*gpuArray(cast(Ukeep,classUnderlying(Q)));
        else
            Q(:,1:k) = Q*Ukeep;
        end
        d = d(1:k);
        c = beta*Ukeep(end,:);
        justRestarted = true;
        sizeV = k+1;
    end

    Ukeep = Usmall(:,ind(1:k0));
    d = d(1:k0);
    if flag_gpu
        U = Q*gpuArray(cast(Ukeep,classUnderlying(Q)));
    else
        U = Q*Ukeep;
    end
end

function r = GramSchmitGEP(V,r,fncB,jj)
    for kk = 1:jj
        tmp = gather_scalar(fncB(V(:,kk))'*r);
        r = r-tmp*V(:,kk);
    end
end
