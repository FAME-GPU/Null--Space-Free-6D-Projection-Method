function [fields,info] = reconstruct_tracked_modes_exp4(Xhost,basis,yee,part,cfg)
%RECONSTRUCT_TRACKED_MODES_EXP4 Joint all-GPU Taylor reconstruction.
    if ~isa(basis.qModes,'gpuArray') || ~isa(yee.r1,'gpuArray')
        error('All-GPU Exp3 requires GPU qModes and Yee coordinates.');
    end
    NF=size(basis.qModes,1);
    nm=size(Xhost,2);
    if size(Xhost,1)~=3*NF, error('Eigenvector dimension is incompatible with qModes.'); end

    Xdev=gpuArray(cast(Xhost,classUnderlying(basis.qModes)));
    geom=build_taylor_geometry_exp4(yee.r1,part.blocks,basis.qModes,cfg.taylor.order);
    pointSets={yee.r1,yee.r2,yee.r3};
    Udev=cell(3,1); compInfo=cell(3,1);
    wait_for_gpu(); total=tic;

    for ell=1:3
        rows=(ell-1)*NF+(1:NF);
        coeffMat=Xdev(rows,:);
        % Staggered sets share local monomials; only centers translate.
        centers=zeros(geom.numCenters,3,'like',pointSets{ell});
        for ib=1:geom.numCenters
            ids=part.blocks{ib};
            R=pointSets{ell}(ids,:);
            centers(ib,:)=(min(R,[],1)+max(R,[],1))/2;
        end
        geomEll=geom; geomEll.centersDevice=centers;
        [u,ci]=reconstruct_component_taylor_batch_exp4(coeffMat,basis,geomEll,cfg.taylor.centerBatchSize);
        Udev{ell}=u;
        compInfo{ell}=ci;
        clear coeffMat u geomEll centers
    end
    wait_for_gpu(Udev{3}); wall=toc(total);

    fields=cell(nm,1);
    for j=1:nm
        fields{j}=struct('u1',reshape(Udev{1}(:,j),yee.szPlus), ...
            'u2',reshape(Udev{2}(:,j),yee.szPlus), ...
            'u3',reshape(Udev{3}(:,j),yee.szPlus));
    end
    info=struct('timeSeconds',wall,'componentInfo',{compInfo}, ...
        'numCenters',part.numCenters,'blocksPerDimension',part.blocksPerDimension, ...
        'rhoBound',part.rhoBound,'numTerms',basis.numTerms);
    clear geom Udev Xdev
end
