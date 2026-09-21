function [fields,info] = reconstruct_tracked_modes_exp4(Xin,basis,yee,part,cfg)
%RECONSTRUCT_TRACKED_MODES_EXP4 GPU Taylor reconstruction without CPU field fallback.
    NF=size(basis.qModes,1);
    nm=size(Xin,2);
    if size(Xin,1)~=3*NF, error('Eigenvector dimension is incompatible with qModes.'); end
    if isa(Xin,'gpuArray')
        Xgpu=Xin;
    else
        Xgpu=gpuArray(cast(Xin,classUnderlying(basis.qModes)));
    end

    geom=build_taylor_geometry_exp4(yee.r1,part.blocks,basis.qModes,cfg.taylor.order);
    pointSets={yee.r1,yee.r2,yee.r3};
    Ugpu=cell(3,1); compInfo=cell(3,1);
    total=tic;
    for ell=1:3
        rows=(ell-1)*NF+(1:NF);
        coeffMat=Xgpu(rows,:);

        centers=zeros(geom.numCenters,3,'like',basis.qModes);
        for ib=1:geom.numCenters
            ids=part.blocks{ib};
            R=pointSets{ell}(ids,:);
            centers(ib,:)=(min(R,[],1)+max(R,[],1))/2;
        end
        geomEll=geom; geomEll.centers=centers;
        [u,ci]=reconstruct_component_taylor_batch_exp4( ...
            coeffMat,basis,geomEll,cfg.taylor.centerBatchSize);
        Ugpu{ell}=u;
        compInfo{ell}=ci;
        clear coeffMat u geomEll centers
    end
    wait_for_gpu(Ugpu{1}); wall=toc(total);

    fields=cell(nm,1);
    for j=1:nm
        fields{j}=struct('u1',reshape(Ugpu{1}(:,j),yee.szPlus), ...
            'u2',reshape(Ugpu{2}(:,j),yee.szPlus), ...
            'u3',reshape(Ugpu{3}(:,j),yee.szPlus));
    end
    info=struct('timeSeconds',wall,'componentInfo',{compInfo}, ...
        'numCenters',part.numCenters,'blocksPerDimension',part.blocksPerDimension, ...
        'rhoBound',part.rhoBound,'numTerms',basis.numTerms,'device','GPU');
    clear geom Ugpu Xgpu
end
