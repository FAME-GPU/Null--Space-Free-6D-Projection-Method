function [uCell,elapsed] = reconstruct_direct_group_exp2b(coeff,qModes,Rcells,blockSize)
%RECONSTRUCT_DIRECT_GROUP_EXP2B Timed blockwise direct dense-phase sum.
%
% Coordinates are supplied by the caller and are not charged to this timer.
% The local coordinate blocks in one timeout-check group are first stacked
% into a small contiguous point list (also outside the timer).  Dense phase
% matrices are then formed in point batches, so peak memory is
% O(blockSize*N_F), independent of N_G.  Stacking avoids artificial batch
% fragmentation at Taylor-center boundaries and keeps the direct timing
% close to the original global blockwise implementation.

    coeff = coeff(:);
    NF = size(qModes,1);
    if numel(coeff) ~= NF
        error('reconstruct_direct_group_exp2b:CoeffMismatch','Coefficient length does not match qModes.');
    end
    blockSize = max(1,round(blockSize));
    nB = numel(Rcells);

    if nB==0
        uCell=cell(0,1); elapsed=0; return;
    end

    counts=zeros(nB,1);
    for ib=1:nB
        counts(ib)=size(Rcells{ib},1);
    end
    Rall=vertcat(Rcells{:});
    wait_for_gpu(Rall);  % coordinate stacking is outside the timer
    M=size(Rall,1);
    uAll=complex(zeros(M,1,'like',coeff));

    wait_for_gpu();
    t0=tic;
    for s=1:blockSize:M
        e=min(M,s+blockSize-1);
        ids=s:e;
        phase=Rall(ids,:)*qModes.';
        uAll(ids)=exp(1i*phase)*coeff;
    end
    wait_for_gpu(uAll);
    elapsed=toc(t0);

    % Split only after the timer, for blockwise error accumulation.
    uCell=cell(nB,1);
    pos=1;
    for ib=1:nB
        last=pos+counts(ib)-1;
        uCell{ib}=uAll(pos:last);
        pos=last+1;
    end
    if pos~=M+1
        error('reconstruct_direct_group_exp2b:SplitBug','Direct output split did not cover all points.');
    end
end
