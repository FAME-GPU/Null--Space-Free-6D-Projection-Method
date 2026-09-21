function geom = build_taylor_geometry_exp4(rPoints,blocks,prototype,p)
%BUILD_TAYLOR_GEOMETRY_EXP4 Block centers/local monomials on the GPU.
    if ~isa(rPoints,'gpuArray') || ~isa(prototype,'gpuArray')
        error('Taylor geometry production path requires GPU point arrays.');
    end

    M=size(rPoints,1);
    nCenters=numel(blocks);
    centerId=zeros(M,1,'uint32');
    centers=zeros(nCenters,3,'like',rPoints);
    Rc=zeros(M,3,'like',rPoints);
    for ib=1:nCenters
        ids=blocks{ib};
        R=rPoints(ids,:);
        rc=(min(R,[],1)+max(R,[],1))/2;
        centers(ib,:)=rc;
        centerId(ids)=ib;
        Rc(ids,:)=R-rc;
    end
    if any(centerId==0), error('Taylor block partition does not cover every point.'); end

    Rx=ones(M,p+1,'like',Rc); Ry=Rx; Rz=Rx;
    for s=1:p
        Rx(:,s+1)=Rx(:,s).*Rc(:,1);
        Ry(:,s+1)=Ry(:,s).*Rc(:,2);
        Rz(:,s+1)=Rz(:,s).*Rc(:,3);
    end
    clear Rc

    geom=struct('centerIdCPU',double(centerId),'centers',centers, ...
        'Rx',Rx,'Ry',Ry,'Rz',Rz,'numPoints',M,'numCenters',nCenters);
end
