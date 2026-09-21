function [KrInvOp,getStats] = make_explicit_krinv_instrumented(sys,tolCG,maxitCG,budget)
%MAKE_EXPLICIT_KRINV_INSTRUMENTED Explicit orthogonalized K_r^{-1} action.
%
% Implements
%   K_r^{-1} = V_1^{-1} M_1 V_1^{-*} - B_r Mhat^{-1} B_r^*
% and instruments the Mhat-solve time separately from the remaining algebra.

    if nargin<4 || isempty(budget), budget=make_run_budget(Inf,Inf); end
    stats = init_stats();
    KrInvOp = @applyOneOrMany;
    getStats = @getStatsLocal;

    function y = applyOneOrMany(q)
        if size(q,2)==1
            y = applyOne(q);
        else
            y = zeros(size(q),'like',q);
            for jj=1:size(q,2)
                y(:,jj)=applyOne(q(:,jj));
            end
        end
    end

    function y = applyOne(q)
        budget.checkCanContinue('Method C inverse entry');
        n=sys.NF;
        if size(q,1)~=2*n
            error('Explicit K_r^{-1} input must have 2*NF rows.');
        end
        if ~gather_scalar(all(isfinite(q(:))))
            error('Explicit K_r^{-1} input contains NaN/Inf.');
        end

        wait_for_gpu(q);
        invTimer=tic;

        r = solveV1H_blockdiag(q, ...
            sys.V11,sys.V12,sys.V21,sys.V22,sys.detV1,n);

        Mcols12r = sys.mass.Mcols12op(r);
        M1r = Mcols12r(1:2*n,:);
        M3Hr = Mcols12r(2*n+1:3*n,:);

        b = M3Hr-sys.WHop(M1r);
        bhat = sys.invCorthDiag.*b;
        if ~gather_scalar(all(isfinite(bhat(:))))
            error('Mhat right-hand side contains NaN/Inf.');
        end

        wait_for_gpu(bhat);
        solveTimer=tic;
        x0=zeros(size(bhat),'like',bhat);
        solveMaxit=budget.capMaxit(maxitCG);
        [u,flag,relres,iter]=pcg(@(x)sys.MhatOp(x), ...
            bhat,tolCG,solveMaxit,[],[],x0);
        wait_for_gpu(u);
        solveElapsed=toc(solveTimer);

        flagCPU=gather_scalar(flag);
        relresCPU=gather_scalar(relres);
        iterCPU=gather_scalar(iter);
        budget.addIterations(iterCPU,'Method C Mhat CG');
        if flagCPU~=0 && relresCPU>10*tolCG
            budget.checkCanContinue('Method C Mhat CG continuation');
        end
        if ~gather_scalar(all(isfinite(u(:)))) || ~isfinite(relresCPU) || ...
                (flagCPU~=0 && relresCPU>10*tolCG)
            error('Mhat CG failed: flag=%d, relres=%.3e, iter=%d.', ...
                flagCPU,relresCPU,iterCPU);
        end

        z = sys.invCorthDiag.*u;
        Mcol3z = sys.mass.Mcol3op(z);
        M3z = Mcol3z(1:2*n,:);

        Wz = sys.Wop(z);
        Mcols12Wz = sys.mass.Mcols12op(Wz);
        M1Wz = Mcols12Wz(1:2*n,:);

        term = M1r-(M3z-M1Wz);
        y = solveV1_blockdiag(term, ...
            sys.V11,sys.V12,sys.V21,sys.V22,sys.detV1,n);

        wait_for_gpu(y);
        invElapsed=toc(invTimer);
        budget.check('Method C inverse return');

        if ~gather_scalar(all(isfinite(y(:))))
            error('Explicit K_r^{-1} output contains NaN/Inf.');
        end

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

    function s=getStatsLocal()
        s=stats;
        s.remainingInverseTime=max(s.totalInverseTime-s.totalMhatSolveTime,0);
        if s.inverseCalls>0
            s.averageInverseTime=s.totalInverseTime/s.inverseCalls;
            s.averageMhatIterations=s.totalMhatIterations/s.inverseCalls;
            s.averageMhatSolveTime=s.totalMhatSolveTime/s.inverseCalls;
        else
            s.averageInverseTime=NaN;
            s.averageMhatIterations=NaN;
            s.averageMhatSolveTime=NaN;
        end
    end
end

function s=init_stats()
    s.inverseCalls=0;
    s.totalInverseTime=0;
    s.totalMhatSolveTime=0;
    s.totalMhatIterations=0;
    s.maxMhatIterations=0;
    s.maxMhatRelativeResidual=0;
    s.flags=zeros(0,1);
    s.iterations=zeros(0,1);
    s.relativeResiduals=zeros(0,1);
    s.solveTimes=zeros(0,1);
    s.inverseTimes=zeros(0,1);
end
