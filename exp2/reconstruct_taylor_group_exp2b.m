function [uCell,elapsed] = reconstruct_taylor_group_exp2b(coeff,qModes,Rcells,rcCells,pre)
%RECONSTRUCT_TAYLOR_GROUP_EXP2B Timed Taylor reconstruction of local blocks.
%
% Coordinate generation is intentionally outside this timer.  Fourier-side
% q-power preprocessing is also outside, matching the panel-(a) timing
% convention.  The timer includes center phases, moments, local monomials,
% and field assembly.

    nB = numel(Rcells);
    if numel(rcCells) ~= nB
        error('reconstruct_taylor_group_exp2b:SizeMismatch','Rcells and rcCells must have the same length.');
    end
    coeff = coeff(:);
    NF = size(qModes,1);
    if numel(coeff) ~= NF
        error('reconstruct_taylor_group_exp2b:CoeffMismatch','Coefficient length does not match qModes.');
    end

    uCell = cell(nB,1);
    wait_for_gpu();
    t0 = tic;
    for ib = 1:nB
        R = Rcells{ib};
        rc = rcCells{ib};
        uCell{ib} = taylor_one_block(coeff,qModes,R,rc,pre);
    end
    if nB>0
        wait_for_gpu(uCell{end});
    else
        wait_for_gpu();
    end
    elapsed = toc(t0);
end

function u = taylor_one_block(coeff,qModes,R,rc,pre)
    p = pre.p;
    alphaList = pre.alphaList;
    L = pre.numTerms;
    bTerms = pre.termBatchSize;
    NF = size(qModes,1);
    mPts = size(R,1);

    Rc = R - rc;
    phaseCenter = exp(1i*(qModes*rc(:)));
    cCenter = coeff .* phaseCenter;

    Rx = ones(mPts,p+1,'like',R);
    Ry = ones(mPts,p+1,'like',R);
    Rz = ones(mPts,p+1,'like',R);
    for s = 1:p
        Rx(:,s+1) = Rx(:,s).*Rc(:,1);
        Ry(:,s+1) = Ry(:,s).*Rc(:,2);
        Rz(:,s+1) = Rz(:,s).*Rc(:,3);
    end

    u = complex(zeros(mPts,1,'like',coeff));
    for s0 = 1:bTerms:L
        s1 = min(L,s0+bTerms-1);
        aa = alphaList(s0:s1,:);
        a1 = aa(:,1).';
        a2 = aa(:,2).';
        a3 = aa(:,3).';
        Qb = pre.Qx(:,a1+1).*pre.Qy(:,a2+1).*pre.Qz(:,a3+1);
        Rb = Rx(:,a1+1).*Ry(:,a2+1).*Rz(:,a3+1);
        mom = Qb.'*cCenter;
        mom = pre.coeffAlpha(s0:s1).*mom;
        u = u + Rb*mom;
    end

    if size(u,1)~=mPts || size(u,2)~=1 || NF~=numel(coeff)
        error('reconstruct_taylor_group_exp2b:InternalShapeBug','Unexpected Taylor output shape.');
    end
end
