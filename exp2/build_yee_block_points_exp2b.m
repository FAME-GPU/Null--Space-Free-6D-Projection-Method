function [R,rc] = build_yee_block_points_exp2b(rangeRow,h,component,prototype)
%BUILD_YEE_BLOCK_POINTS_EXP2B Generate one staggered Yee block on demand.
%
% No global Yee coordinate array is formed.  rangeRow is
% [i0 i1 j0 j1 k0 k1] in integer Yee-cell indices.

    if numel(rangeRow) ~= 6
        error('build_yee_block_points_exp2b:BadRange','rangeRow must have six entries.');
    end
    if component < 1 || component > 3 || component ~= round(component)
        error('build_yee_block_points_exp2b:BadComponent','component must be 1, 2, or 3.');
    end

    rr = double(rangeRow(:).');
    i0=rr(1); i1=rr(2); j0=rr(3); j1=rr(4); k0=rr(5); k1=rr(6);
    if i1<i0 || j1<j0 || k1<k0
        error('build_yee_block_points_exp2b:BadRange','Invalid index range.');
    end

    iv = cast((i0:i1).','like',prototype);
    jv = cast((j0:j1).','like',prototype);
    kv = cast((k0:k1).','like',prototype);
    [I,J,K] = ndgrid(iv,jv,kv);

    switch component
        case 1
            R = h*[I(:)+0.5, J(:),     K(:)];
            rcCPU = h*[(i0+i1)/2+0.5, (j0+j1)/2,     (k0+k1)/2];
        case 2
            R = h*[I(:),     J(:)+0.5, K(:)];
            rcCPU = h*[(i0+i1)/2,     (j0+j1)/2+0.5, (k0+k1)/2];
        case 3
            R = h*[I(:),     J(:),     K(:)+0.5];
            rcCPU = h*[(i0+i1)/2,     (j0+j1)/2,     (k0+k1)/2+0.5];
    end
    rc = cast(rcCPU,'like',prototype);
end
