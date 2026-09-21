function [KrInvOp,getStats] = make_explicit_krinv_compact(sys,tolCG,maxitCG,budget)
%MAKE_EXPLICIT_KRINV_COMPACT Low-memory scalar-isotropic explicit K_r^{-1}.
%
% Implements the same orthogonalized explicit inverse as the manuscript,
% specialized to M=diag(Ms,Ms,Ms), hence M3=0.  All scalar mass actions are
% performed serially through the common exact rank-1 backend.

    if nargin<4 || isempty(budget), budget=make_run_budget(Inf,Inf); end
    stats=init_stats();
    KrInvOp=@applyOneOrMany; getStats=@getStatsLocal;
    n=sys.NF; mass=sys.mass;
    V11=sys.V11; V12=sys.V12; V21=sys.V21; V22=sys.V22; detV1=sys.detV1;
    W1=sys.W1; W2=sys.W2; invC=sys.invCorthDiag;

    function y=applyOneOrMany(q)
        if size(q,2)==1
            y=applyOne(q);
        else
            y=zeros(size(q),'like',q);
            for jj=1:size(q,2), y(:,jj)=applyOne(q(:,jj)); end
        end
    end

    function y=applyOne(q)
        budget.checkCanContinue('Method C inverse entry');
        if size(q,1)~=2*n, error('Explicit K_r^{-1} input must have 2*NF rows.'); end
        if ~gather_scalar(all(isfinite(q(:)))), error('Explicit K_r^{-1} input contains NaN/Inf.'); end
        wait_for_gpu(q); invTimer=tic;

        r=solveV1H_blockdiag(q,V11,V12,V21,V22,detV1,n);
        mr1=mass.MscalarOp(r(1:n));
        mr2=mass.MscalarOp(r(n+1:2*n));

        b=-(conj(W1).*mr1 + conj(W2).*mr2);
        bhat=invC.*b;
        clear b
        if ~gather_scalar(all(isfinite(bhat(:)))), error('Mhat right-hand side contains NaN/Inf.'); end

        wait_for_gpu(bhat); solveTimer=tic;
        x0=zeros(size(bhat),'like',bhat);
        solveMaxit=budget.capMaxit(maxitCG);
        try
            [u,flag,relres,iter]=pcg(@applyMhatBudgeted,bhat,tolCG,solveMaxit,[],[],x0);
        catch MEpcg
            rethrow_exp1_controlled(MEpcg,'Method C Mhat CG',budget);
        end
        wait_for_gpu(u); solveElapsed=toc(solveTimer);

        flagCPU=double(gather_scalar(flag)); relresCPU=double(gather_scalar(relres));
        iterCPU=double(gather_scalar(iter));
        budget.addIterations(iterCPU,'Method C Mhat CG');
        if flagCPU~=0 && relresCPU>10*tolCG
            budget.checkCanContinue('Method C Mhat CG continuation');
        end
        if ~gather_scalar(all(isfinite(u(:)))) || ~isfinite(relresCPU) || ...
                (flagCPU~=0 && relresCPU>10*tolCG)
            error('Mhat CG failed: flag=%d, relres=%.3e, iter=%d.',flagCPU,relresCPU,iterCPU);
        end

        z=invC.*u; clear u bhat x0
        t=mass.MscalarOp(W1.*z); mr1=mr1+t; clear t
        t=mass.MscalarOp(W2.*z); mr2=mr2+t; clear t z
        term=[mr1;mr2]; clear mr1 mr2
        y=solveV1_blockdiag(term,V11,V12,V21,V22,detV1,n); clear term r

        wait_for_gpu(y); invElapsed=toc(invTimer); budget.check('Method C inverse return');
        if ~gather_scalar(all(isfinite(y(:)))), error('Explicit K_r^{-1} output contains NaN/Inf.'); end

        stats.inverseCalls=stats.inverseCalls+1;
        stats.totalInverseTime=stats.totalInverseTime+invElapsed;
        stats.totalMhatSolveTime=stats.totalMhatSolveTime+solveElapsed;
        stats.totalMhatIterations=stats.totalMhatIterations+iterCPU;
        stats.maxMhatIterations=max(stats.maxMhatIterations,iterCPU);
        stats.maxMhatRelativeResidual=max(stats.maxMhatRelativeResidual,relresCPU);
        stats.flags(end+1,1)=flagCPU; %#ok<AGROW>
        stats.iterations(end+1,1)=iterCPU; %#ok<AGROW>
        stats.relativeResiduals(end+1,1)=relresCPU; %#ok<AGROW>
        stats.solveTimes(end+1,1)=solveElapsed; %#ok<AGROW>
        stats.inverseTimes(end+1,1)=invElapsed; %#ok<AGROW>
    end


    function y=applyMhatBudgeted(x)
        budget.checkCanContinue('Method C Mhat matvec');
        y=apply_Mhat_C_scalar_compact(x,mass,W1,W2,invC);
    end

    function s=getStatsLocal()
        s=stats; s.remainingInverseTime=max(s.totalInverseTime-s.totalMhatSolveTime,0);
        if s.inverseCalls>0
            s.averageInverseTime=s.totalInverseTime/s.inverseCalls;
            s.averageMhatIterations=s.totalMhatIterations/s.inverseCalls;
            s.averageMhatSolveTime=s.totalMhatSolveTime/s.inverseCalls;
        else
            s.averageInverseTime=NaN; s.averageMhatIterations=NaN; s.averageMhatSolveTime=NaN;
        end
    end
end

function s=init_stats()
    s.inverseCalls=0; s.totalInverseTime=0; s.totalMhatSolveTime=0;
    s.totalMhatIterations=0; s.maxMhatIterations=0; s.maxMhatRelativeResidual=0;
    s.flags=zeros(0,1); s.iterations=zeros(0,1); s.relativeResiduals=zeros(0,1);
    s.solveTimes=zeros(0,1); s.inverseTimes=zeros(0,1);
end
