function [KrInvOp,getStats] = make_nested_krinv_instrumented(sys,cfg,tolKr,tolM,maxitKr,maxitM,budget)
%MAKE_NESTED_KRINV_INSTRUMENTED Nested realization of K_r^{-1}.
%
% For each input b, solve K_r x=b by PCG.  Every matrix-vector product
%
%       K_r z = V^* M^{-1} V z
%
% contains a second PCG solve with M.  The two tolerances are deliberately
% independent:
%       tolKr : outer PCG tolerance for K_r x=b;
%       tolM  : inner PCG tolerance used in every M^{-1} action.
%
% The large-scale N=5 production setting uses tolKr=1e-10 and tolM=1e-12.

    if nargin<7 || isempty(budget), budget=make_run_budget(Inf,Inf); end
    stats = init_stats();
    stats.tolKr = tolKr;
    stats.tolM = tolM;
    KrInvOp = @applyOneOrMany;
    getStats = @getStatsLocal;

    function y = applyOneOrMany(b)
        if size(b,2)==1
            y = applyOne(b);
        else
            y = zeros(size(b),'like',b);
            for jj=1:size(b,2)
                y(:,jj)=applyOne(b(:,jj));
            end
        end
    end

    function x = applyOne(b)
        budget.checkCanContinue('Method B K_r solve entry');
        if size(b,1)~=sys.nr
            error('Nested K_r^{-1} input must have 2*NF rows.');
        end
        if ~gather_scalar(all(isfinite(b(:))))
            error('Nested K_r^{-1} input contains NaN/Inf.');
        end

        wait_for_gpu(b);
        invTimer = tic;

        x0 = zeros(size(b),'like',b);
        if cfg.nested.useKrPreconditioner
            M1 = @(r) cfg.material.epsC*(r./sys.sigmaR);
        else
            M1 = [];
        end

        outerMaxit = budget.capMaxit(maxitKr);
        try
            [x,flag,relres,iter] = pcg(@applyKr,b,tolKr,outerMaxit,M1,[],x0);
        catch MEpcg
            rethrow_exp1_controlled(MEpcg,'Method B outer K_r CG',budget);
        end
        wait_for_gpu(x);
        invElapsed = toc(invTimer);

        flagCPU = gather_scalar(flag);
        relresCPU = gather_scalar(relres);
        iterCPU = gather_scalar(iter);
        budget.addIterations(iterCPU,'Method B outer K_r CG');
        if flagCPU~=0 && relresCPU>10*tolKr
            budget.checkCanContinue('Method B outer K_r CG continuation');
        end
        if ~gather_scalar(all(isfinite(x(:)))) || ~isfinite(relresCPU) || ...
                (flagCPU~=0 && relresCPU>10*tolKr)
            error('Nested K_r solve failed: flag=%d, relres=%.3e, iter=%d.', ...
                flagCPU,relresCPU,iterCPU);
        end

        stats.inverseCalls = stats.inverseCalls+1;
        stats.totalInverseTime = stats.totalInverseTime+invElapsed;
        stats.totalKrIterations = stats.totalKrIterations+iterCPU;
        stats.maxKrIterations = max(stats.maxKrIterations,iterCPU);
        stats.maxKrRelativeResidual = max(stats.maxKrRelativeResidual,relresCPU);
        stats.krFlags(end+1,1)=flagCPU; %#ok<AGROW>
        stats.krIterations(end+1,1)=iterCPU; %#ok<AGROW>
        stats.krRelativeResiduals(end+1,1)=relresCPU; %#ok<AGROW>
        stats.inverseTimes(end+1,1)=invElapsed; %#ok<AGROW>
    end

    function y = applyKr(z)
        budget.checkCanContinue('Method B inner M solve entry');
        Vz = apply_V_blockdiag(z,sys.Vdata);

        wait_for_gpu(Vz);
        mTimer = tic;
        y0 = zeros(size(Vz),'like',Vz);
        innerMaxit = budget.capMaxit(maxitM);
        try
            [MyInv,flagM,relresM,iterM] = pcg( ...
                @applyMassBudgeted,Vz,tolM,innerMaxit,[],[],y0);
        catch MEpcg
            rethrow_exp1_controlled(MEpcg,'Method B inner M CG',budget);
        end
        wait_for_gpu(MyInv);
        mElapsed = toc(mTimer);

        flagCPU = gather_scalar(flagM);
        relresCPU = gather_scalar(relresM);
        iterCPU = gather_scalar(iterM);
        budget.addIterations(iterCPU,'Method B inner M CG');
        if flagCPU~=0 && relresCPU>10*tolM
            budget.checkCanContinue('Method B inner M CG continuation');
        end
        if ~gather_scalar(all(isfinite(MyInv(:)))) || ~isfinite(relresCPU) || ...
                (flagCPU~=0 && relresCPU>10*tolM)
            error('Nested M solve failed: flag=%d, relres=%.3e, iter=%d.', ...
                flagCPU,relresCPU,iterCPU);
        end

        stats.massSolveCalls = stats.massSolveCalls+1;
        stats.totalMassSolveTime = stats.totalMassSolveTime+mElapsed;
        stats.totalMassIterations = stats.totalMassIterations+iterCPU;
        stats.maxMassIterations = max(stats.maxMassIterations,iterCPU);
        stats.maxMassRelativeResidual = max(stats.maxMassRelativeResidual,relresCPU);
        stats.massFlags(end+1,1)=flagCPU; %#ok<AGROW>
        stats.massIterations(end+1,1)=iterCPU; %#ok<AGROW>
        stats.massRelativeResiduals(end+1,1)=relresCPU; %#ok<AGROW>
        stats.massSolveTimes(end+1,1)=mElapsed; %#ok<AGROW>

        y = apply_VH_blockdiag(MyInv,sys.Vdata);
    end


    function y = applyMassBudgeted(u)
        budget.checkCanContinue('Method B inner M matvec');
        y = sys.mass.Mop(u);
    end

    function s = getStatsLocal()
        s = stats;
        s.remainingInverseTime = max(s.totalInverseTime-s.totalMassSolveTime,0);
        if s.inverseCalls>0
            s.averageInverseTime = s.totalInverseTime/s.inverseCalls;
            s.averageKrIterations = s.totalKrIterations/s.inverseCalls;
        else
            s.averageInverseTime = NaN;
            s.averageKrIterations = NaN;
        end
        if s.massSolveCalls>0
            s.averageMassIterations = s.totalMassIterations/s.massSolveCalls;
            s.averageMassSolveTime = s.totalMassSolveTime/s.massSolveCalls;
        else
            s.averageMassIterations = NaN;
            s.averageMassSolveTime = NaN;
        end
    end
end

function s = init_stats()
    s.inverseCalls=0;
    s.totalInverseTime=0;
    s.totalKrIterations=0;
    s.maxKrIterations=0;
    s.maxKrRelativeResidual=0;
    s.krFlags=zeros(0,1);
    s.krIterations=zeros(0,1);
    s.krRelativeResiduals=zeros(0,1);
    s.inverseTimes=zeros(0,1);

    s.massSolveCalls=0;
    s.totalMassSolveTime=0;
    s.totalMassIterations=0;
    s.maxMassIterations=0;
    s.maxMassRelativeResidual=0;
    s.massFlags=zeros(0,1);
    s.massIterations=zeros(0,1);
    s.massRelativeResiduals=zeros(0,1);
    s.massSolveTimes=zeros(0,1);
end
