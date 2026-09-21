function part = build_visual_plane_partition_exp4(plane,basis,cfg)
%BUILD_VISUAL_PLANE_PARTITION_EXP4 Automatic 2D block partition by phase radius.
% Uses the exact effective two-coordinate frequencies for each plane.
    m=plane.sz(1);
    if plane.sz(2)~=m, error('Visual plane must be square.'); end
    q=basis.qModes;
    switch lower(plane.kind)
        case 'z0'
            q1=q(:,1); q2=q(:,2);
        case 'x0'
            q1=q(:,2); q2=q(:,3);
        case 'xy'
            q1=q(:,1)+q(:,2); q2=q(:,3);
        otherwise
            error('Unknown plane kind.');
    end
    q2normMax=gather_scalar(max(abs(q1)+abs(q2)));
    dx=2*cfg.visual.halfWidth/(m-1);
    target=cfg.visual.phaseRadiusTarget;
    chosen=[];
    for nb=1:cfg.visual.maxBlocksPerDimension
        maxLen=ceil(m/nb);
        halfSpan=max(0,(maxLen-1)*dx/2);
        rho=q2normMax*halfSpan;
        if rho<=target*(1+cfg.taylor.phaseRadiusSlack)
            chosen=nb; break;
        end
    end
    if isempty(chosen)
        error('No visual-plane partition satisfies phase-radius target.');
    end
    groups=split_dim(m,chosen);
    blocks=cell(chosen^2,1); c=0;
    for j=1:chosen
        for i=1:chosen
            c=c+1;
            [I,J]=ndgrid(groups{i},groups{j});
            blocks{c}=sub2ind([m m],I(:),J(:));
        end
    end
    maxLen=ceil(m/chosen);
    halfSpan=max(0,(maxLen-1)*dx/2);
    part=struct('blocks',{blocks},'blocksPerDimension',chosen, ...
        'numCenters',chosen^2,'rhoBound',q2normMax*halfSpan, ...
        'effectiveQ1NormMax',q2normMax);
end

function groups=split_dim(m,nBlocks)
    base=floor(m/nBlocks); remn=mod(m,nBlocks);
    groups=cell(nBlocks,1); s=1;
    for b=1:nBlocks
        len=base+(b<=remn);
        groups{b}=s:(s+len-1); s=s+len;
    end
end
