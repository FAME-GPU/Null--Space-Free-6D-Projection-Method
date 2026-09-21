function out = reconstruct_visual_plane_exp4(Xin,basis,plane,part,cfg)
%RECONSTRUCT_VISUAL_PLANE_EXP4 GPU reconstruction of |E|^2 on one plane.
    NF=size(basis.qModes,1);
    if size(Xin,1)~=3*NF || size(Xin,2)~=1
        error('Xin must be a single 3*NF Fourier eigenvector.');
    end
    if isa(Xin,'gpuArray')
        Xgpu=Xin;
    else
        Xgpu=gpuArray(cast(Xin,classUnderlying(basis.qModes)));
    end

    geom=build_taylor_geometry_exp4(plane.points,part.blocks,basis.qModes,cfg.taylor.order);
    U=cell(3,1); compInfo=cell(3,1);
    t=tic;
    for ell=1:3
        rows=(ell-1)*NF+(1:NF);
        coeff=Xgpu(rows);
        [u,ci]=reconstruct_component_taylor_batch_exp4( ...
            coeff,basis,geom,cfg.visual.centerBatchSize);
        U{ell}=u;
        compInfo{ell}=ci;
        clear coeff u
    end
    wait_for_gpu(U{1}); wall=toc(t);

    I=abs(U{1}).^2+abs(U{2}).^2+abs(U{3}).^2;
    I=reshape(I,plane.sz);
    out=struct();
    out.kind=plane.kind;
    out.axis1=plane.axis1CPU;
    out.axis2=plane.axis2CPU;
    out.axisLabels=plane.labels;
    out.intensityRaw=I;
    out.numCenters=part.numCenters;
    out.blocksPerDimension=part.blocksPerDimension;
    out.rhoBound=part.rhoBound;
    out.timeSeconds=wall;
    out.componentInfo=compInfo;
end
