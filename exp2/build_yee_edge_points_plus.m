function [r1_plus,r2_plus,r3_plus,gridInfo] = ...
    build_yee_edge_points_plus(nYeeCrop,hYee,prototype)
%BUILD_YEE_EDGE_POINTS_PLUS Build Yee points directly on the GPU.
%
% prototype is a scalar/array on the desired device and precision.

    if nargin < 3 || isempty(prototype)
        prototype = 0;
    end

    idxPlusCPU = (-(nYeeCrop+1):(nYeeCrop+1)).';
    idxPlus = cast(idxPlusCPU,'like',prototype);
    [I,J,K] = ndgrid(idxPlus,idxPlus,idxPlus);

    r1_plus = hYee*[I(:)+0.5,J(:),K(:)];
    r2_plus = hYee*[I(:),J(:)+0.5,K(:)];
    r3_plus = hYee*[I(:),J(:),K(:)+0.5];

    idxInnerCPU = (-nYeeCrop:nYeeCrop).';
    idxInner = cast(idxInnerCPU,'like',prototype);

    gridInfo.nYeeCrop = nYeeCrop;
    gridInfo.hYee = hYee;
    gridInfo.idxPlus = idxPlus;
    gridInfo.szPlus = size(I);
    gridInfo.numPlus = numel(I);
    gridInfo.idxInner = idxInner;
    gridInfo.szInner = [numel(idxInnerCPU),numel(idxInnerCPU),numel(idxInnerCPU)];
    gridInfo.innerRangeInPlus = 2:(numel(idxPlusCPU)-1);
    gridInfo.onGPU = isa(idxPlus,'gpuArray');
    if gridInfo.onGPU
        gridInfo.precision = classUnderlying(idxPlus);
    else
        gridInfo.precision = class(idxPlus);
    end
end
