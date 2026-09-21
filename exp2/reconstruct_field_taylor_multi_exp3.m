function [u,info] = reconstruct_field_taylor_multi_exp3(coeff,qModes,rPoints,p,termBatchSize,blocks)
%RECONSTRUCT_FIELD_TAYLOR_MULTI_EXP3 64-center paper Taylor reconstruction.
    if nargin<6 || isempty(blocks)
        error('A precomputed Yee block partition is required.');
    end
    [u,baseInfo]=taylor_on_blocks_exp3(coeff,qModes,rPoints,p,termBatchSize,blocks,[]);
    info=baseInfo;
    info.method='multi-center Taylor';
    info.numCenters=numel(blocks);
end
