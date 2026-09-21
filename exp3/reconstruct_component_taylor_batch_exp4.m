function [u,info] = reconstruct_component_taylor_batch_exp4(coeffMat,basis,geom,centerBatchSize)
%RECONSTRUCT_COMPONENT_TAYLOR_BATCH_EXP4 All-GPU multi-center/multi-RHS.
    if ~isa(coeffMat,'gpuArray') || ~isa(basis.qModes,'gpuArray')
        error('All-GPU reconstruction requires gpuArray coefficients and qModes.');
    end
    coeffMat=reshape(coeffMat,size(coeffMat,1),[]);
    NF=size(basis.qModes,1); nrhs=size(coeffMat,2);
    if size(coeffMat,1)~=NF, error('Coefficient row count does not match qModes.'); end
    L=basis.numTerms; C=geom.numCenters;
    centerBatchSize=max(1,min(C,round(centerBatchSize)));

    moments=complex(zeros(L,C,nrhs,'like',coeffMat));
    wait_for_gpu(); tMoment=tic;
    for c0=1:centerBatchSize:C
        c1=min(C,c0+centerBatchSize-1);
        ids=c0:c1; cb=numel(ids);
        centers=geom.centersDevice(ids,:);
        phase=exp(1i*(basis.qModes*centers.'));
        tmp=reshape(phase,[NF cb 1]).*reshape(coeffMat,[NF 1 nrhs]);
        cStack=reshape(tmp,[NF cb*nrhs]);
        momStack=basis.Qalpha.'*cStack;
        momStack=basis.coeffAlpha.*momStack;
        moments(:,ids,:)=reshape(momStack,[L cb nrhs]);
        clear centers phase tmp cStack momStack
    end
    wait_for_gpu(moments); momentTime=toc(tMoment);

    M=geom.numPoints;
    u=complex(zeros(M,nrhs,'like',coeffMat));
    centerId=geom.centerIdHost;
    aList=basis.alphaList;
    wait_for_gpu(); tEval=tic;
    for ia=1:L
        a=aList(ia,:);
        phi=geom.Rx(:,a(1)+1).*geom.Ry(:,a(2)+1).*geom.Rz(:,a(3)+1);
        mval=reshape(moments(ia,centerId,:),[M nrhs]);
        u=u+phi.*mval;
    end
    wait_for_gpu(u); evalTime=toc(tEval);
    info=struct('momentTimeSeconds',momentTime,'evaluationTimeSeconds',evalTime, ...
        'totalTimeSeconds',momentTime+evalTime,'numCenters',C,'numTerms',L, ...
        'centerBatchSize',centerBatchSize);
end
