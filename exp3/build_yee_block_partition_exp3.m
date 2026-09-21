function blocks = build_yee_block_partition_exp3(szPlus,nBlocks)
%BUILD_YEE_BLOCK_PARTITION_EXP3 4x4x4 nearly equal contiguous grid blocks.
    if numel(szPlus)~=3 || any(szPlus~=szPlus(1))
        error('Expected a cubic plus-grid size.');
    end
    m=szPlus(1);
    groups=split_one_dimension(m,nBlocks);
    blocks=cell(nBlocks^3,1);
    c=0;
    for k=1:nBlocks
        for j=1:nBlocks
            for i=1:nBlocks
                c=c+1;
                [I,J,K]=ndgrid(groups{i},groups{j},groups{k});
                blocks{c}=sub2ind(szPlus,I(:),J(:),K(:));
            end
        end
    end
end

function groups=split_one_dimension(m,nBlocks)
    base=floor(m/nBlocks);
    remn=mod(m,nBlocks);
    groups=cell(nBlocks,1);
    s=1;
    for b=1:nBlocks
        len=base+(b<=remn);
        groups{b}=s:(s+len-1);
        s=s+len;
    end
end
