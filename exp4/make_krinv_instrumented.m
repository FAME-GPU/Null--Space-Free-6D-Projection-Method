function [KrInvOp,getStats] = make_krinv_instrumented(sys,tolCG,maxitCG)
%MAKE_KRINV_INSTRUMENTED Explicit K_r^{-1} action with detailed timing.
%
% The statistics are used only for diagnostics/storage in Experiment 3(b).
% The final donut plot uses the top-level eigensolver time, reconstruction
% time, and other preparation time; the inner timing breakdown is saved for
% later analysis.

    stats.calls = 0;
    stats.totalIterations = 0;
    stats.maxIterations = 0;
    stats.maxRelativeResidual = 0;
    stats.flags = zeros(0,1);
    stats.iterations = zeros(0,1);
    stats.relativeResiduals = zeros(0,1);
    stats.totalActionSeconds = 0;
    stats.totalInnerCGSeconds = 0;

    KrInvOp = @applyOneOrMany;
    getStats = @getStatsLocal;

    function y = applyOneOrMany(q)
        if size(q,2)==1
            y = applyOne(q);
        else
            y = complex(zeros(size(q),'like',q));
            for jj=1:size(q,2)
                y(:,jj)=applyOne(q(:,jj));
            end
        end
    end

    function y = applyOne(q)
        n=sys.NF;
        if size(q,1)~=2*n
            error('make_krinv_instrumented:BadInput','q must have 2*NF rows.');
        end
        if ~gather_scalar(all(isfinite(q(:))))
            error('make_krinv_instrumented:NonFiniteInput','KrInv input is non-finite.');
        end

        wait_for_gpu();
        tAction=tic;

        r = solveV1H_blockdiag(q, ...
            sys.V11,sys.V12,sys.V21,sys.V22,sys.detV1,n);

        Mcols12r = sys.mass.Mcols12op(r);
        M1r = Mcols12r(1:2*n,:);
        M3Hr = Mcols12r(2*n+1:3*n,:);

        b = M3Hr-sys.WHop(M1r);
        bhat = sys.invCorthDiag.*b;
        if ~gather_scalar(all(isfinite(bhat(:))))
            error('make_krinv_instrumented:NonFiniteRHS', ...
                'Inner-CG right-hand side contains NaN/Inf.');
        end

        x0 = zeros(size(bhat),'like',bhat);
        wait_for_gpu();
        tCG=tic;
        [u,flag,relres,iter] = pcg(@(x)sys.MhatOp(x), ...
            bhat,tolCG,maxitCG,[],[],x0);
        wait_for_gpu(u);
        cgSeconds=toc(tCG);

        flagCPU=gather_scalar(flag);
        relresCPU=gather_scalar(relres);
        iterCPU=gather_scalar(iter);

        finiteU=gather_scalar(all(isfinite(u(:))));
        if ~finiteU || ~isfinite(relresCPU) || ...
                (flagCPU~=0 && relresCPU>10*tolCG)
            error('make_krinv_instrumented:InnerCGFailure', ...
                'Inner Mhat CG failed: flag=%d, relres=%.3e, iter=%d.', ...
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

        if ~gather_scalar(all(isfinite(y(:))))
            error('make_krinv_instrumented:NonFiniteOutput','KrInv output is non-finite.');
        end

        wait_for_gpu(y);
        actionSeconds=toc(tAction);

        stats.calls=stats.calls+1;
        stats.totalIterations=stats.totalIterations+iterCPU;
        stats.maxIterations=max(stats.maxIterations,iterCPU);
        stats.maxRelativeResidual=max(stats.maxRelativeResidual,relresCPU);
        stats.flags(end+1,1)=flagCPU; %#ok<AGROW>
        stats.iterations(end+1,1)=iterCPU; %#ok<AGROW>
        stats.relativeResiduals(end+1,1)=relresCPU; %#ok<AGROW>
        stats.totalActionSeconds=stats.totalActionSeconds+actionSeconds;
        stats.totalInnerCGSeconds=stats.totalInnerCGSeconds+cgSeconds;
    end

    function s = getStatsLocal()
        s=stats;
        if stats.calls>0
            s.averageIterations=stats.totalIterations/stats.calls;
            s.averageActionSeconds=stats.totalActionSeconds/stats.calls;
            s.averageInnerCGSeconds=stats.totalInnerCGSeconds/stats.calls;
        else
            s.averageIterations=NaN;
            s.averageActionSeconds=NaN;
            s.averageInnerCGSeconds=NaN;
        end
        s.totalInverseActionOtherSeconds=max(0, ...
            stats.totalActionSeconds-stats.totalInnerCGSeconds);
    end
end
