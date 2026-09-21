function [u,info] = reconstruct_field_taylor_single_exp3(coeff,qModes,rPoints,p,termBatchSize,r0)
%RECONSTRUCT_FIELD_TAYLOR_SINGLE_EXP3 Paper single-center Taylor formula.
%
% No q-mode recentering is used.  For the manuscript experiment r0=0.
    if nargin<6 || isempty(r0), r0=[0 0 0]; end
    [u,baseInfo]=taylor_on_blocks_exp3(coeff,qModes,rPoints,p,termBatchSize,{(1:size(rPoints,1)).'},r0);
    info=baseInfo;
    info.method='single-center Taylor';
    info.numCenters=1;
end
