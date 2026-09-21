function plane=build_visual_plane_exp4(kind,cfg,prototype)
%BUILD_VISUAL_PLANE_EXP4 Build regular physical plane directly on the GPU.
    if nargin<3 || ~isa(prototype,'gpuArray')
        error('Visual-plane construction requires a GPU prototype.');
    end
    m=cfg.visual.numPoints;
    if mod(m,2)==0, error('cfg.visual.numPoints must be odd.'); end
    sCPU=linspace(-cfg.visual.halfWidth,cfg.visual.halfWidth,m);
    s=gpuArray(cast(sCPU,classUnderlying(prototype)));
    [A,B]=ndgrid(s,s);
    z=zeros(numel(A),1,'like',A);
    switch lower(kind)
        case 'z0'
            points=[A(:),B(:),z];
            labels={'x','y'};
        case 'x0'
            points=[z,A(:),B(:)];
            labels={'y','z'};
        case 'xy'
            points=[A(:),A(:),B(:)];
            labels={'s','z'};
        otherwise
            error('Unknown visual plane: %s',kind);
    end
    plane=struct('kind',kind,'points',points,'sz',[m m], ...
        'axis1CPU',sCPU,'axis2CPU',sCPU,'labels',{labels});
end
