function part = build_auto_taylor_partition_light_exp3(n,h,basis,cfg)
%BUILD_AUTO_TAYLOR_PARTITION_LIGHT_EXP3 Lightweight regular-grid partition.
%
% This variant is used for the large-window study.  It stores only the
% one-dimensional contiguous index groups and avoids constructing all 3D
% block-index vectors or all Yee point coordinates.

    m = 2*n + 3;
    target = cfg.taylor.phaseRadiusTarget;
    chosen = [];
    rhoBound = NaN;
    maxLen = NaN;

    for nb = 1:cfg.taylor.maxBlocksPerDimension
        maxLenTry = ceil(m/nb);
        halfSpan = max(0,(maxLenTry-1)*h/2);
        rhoTry = basis.qL1Max*halfSpan;
        if rhoTry <= target*(1+cfg.taylor.phaseRadiusSlack)
            chosen = nb;
            rhoBound = rhoTry;
            maxLen = maxLenTry;
            break
        end
    end

    if isempty(chosen)
        error(['No Taylor partition up to %d^3 centers satisfies rho<=%.3f ', ...
            '(m=%d, h=%g, qL1Max=%g).'], ...
            cfg.taylor.maxBlocksPerDimension,target,m,h,basis.qL1Max);
    end

    groups = split_one_dimension(m,chosen);
    part = struct();
    part.groups = groups;
    part.blocksPerDimension = chosen;
    part.numCenters = chosen^3;
    part.rhoBound = rhoBound;
    part.maxGroupLength = maxLen;
    part.halfSpanBound = max(0,(maxLen-1)*h/2);
    part.szPlus = [m m m];
    part.n = n;
    part.h = h;
end

function groups = split_one_dimension(m,nBlocks)
    base = floor(m/nBlocks);
    remn = mod(m,nBlocks);
    groups = cell(nBlocks,1);
    s = 1;
    for b = 1:nBlocks
        len = base + (b<=remn);
        groups{b} = s:(s+len-1);
        s = s + len;
    end
    if s ~= m+1
        error('Internal partition error: groups do not cover the full dimension.');
    end
end
