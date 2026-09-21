function part = build_auto_taylor_partition_exp4(yeePts,basis,cfg)
%BUILD_AUTO_TAYLOR_PARTITION_EXP4 Conservative phase-radius partition.
%
% For a centered rectangular block with half-spans h_j,
% max_{r in block}|q^T(r-r_c)| = sum_j |q_j| h_j.  The cubic Yee
% partition therefore admits the conservative bound
% rho <= max_q ||q||_1 * h_half,max.

    m=yeePts.szPlus(1);
    h=yeePts.h;
    target=cfg.taylor.phaseRadiusTarget;
    chosen=[];
    for nb=1:cfg.taylor.maxBlocksPerDimension
        maxLen=ceil(m/nb);
        halfSpan=max(0,(maxLen-1)*h/2);
        rhoBound=basis.qL1Max*halfSpan;
        if rhoBound<=target*(1+cfg.taylor.phaseRadiusSlack)
            chosen=nb;
            break
        end
    end
    if isempty(chosen)
        error(['No Taylor partition up to %d^3 centers satisfies rho<=%.3f ', ...
            '(m=%d, h=%g, qL1Max=%g).'], ...
            cfg.taylor.maxBlocksPerDimension,target,m,h,basis.qL1Max);
    end

    blocks=build_yee_block_partition_exp3(yeePts.szPlus,chosen);
    maxLen=ceil(m/chosen);
    halfSpan=max(0,(maxLen-1)*h/2);
    rhoBound=basis.qL1Max*halfSpan;
    part=struct('blocks',{blocks},'blocksPerDimension',chosen, ...
        'numCenters',chosen^3,'rhoBound',rhoBound,'maxGroupLength',maxLen, ...
        'halfSpanBound',halfSpan);
end
