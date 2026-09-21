function [u,info] = taylor_on_blocks_exp3(coeff,qModes,rPoints,p,termBatchSize,blocks,fixedCenter)
%TAYLOR_ON_BLOCKS_EXP3 Separated Taylor evaluation on one or many blocks.
%
% u(r) = sum_j c_j exp(i q_j^T r)
%      ~= sum_|a|<=p i^|a|/a! (r-r_c)^a gamma_a(r_c),
% gamma_a(r_c)=sum_j c_j exp(i q_j^T r_c) q_j^a.

    coeff=coeff(:);
    NF=size(qModes,1); M=size(rPoints,1);
    if numel(coeff)~=NF, error('Coefficient length does not match qModes.'); end
    if nargin<5 || isempty(termBatchSize), termBatchSize=32; end

    alphaList=build_multi_index_list_3d_exp3(p);
    L=size(alphaList,1);
    termBatchSize=max(1,min(L,round(termBatchSize)));

    % Precompute q powers once; they are reused by all centers.
    Qx=ones(NF,p+1,'like',qModes); Qy=Qx; Qz=Qx;
    for s=1:p
        Qx(:,s+1)=Qx(:,s).*qModes(:,1);
        Qy(:,s+1)=Qy(:,s).*qModes(:,2);
        Qz(:,s+1)=Qz(:,s).*qModes(:,3);
    end

    coeffAlphaCPU=zeros(L,1);
    for ia=1:L
        a=alphaList(ia,:);
        coeffAlphaCPU(ia)=(1i)^sum(a)*exp(-sum(gammaln(a+1)));
    end
    coeffAlpha=cast(coeffAlphaCPU,'like',coeff);

    u=complex(zeros(M,1,'like',coeff));
    rhoMax=0;
    maxBlockPts=0;
    maxCenterAux=0;
    wait_for_gpu(); t0=tic;

    for ib=1:numel(blocks)
        idsCPU=blocks{ib};
        ids=idsCPU;
        R=rPoints(ids,:);
        if isempty(fixedCenter)
            rmin=min(R,[],1); rmax=max(R,[],1);
            rc=(rmin+rmax)/2;
        else
            rc=cast(fixedCenter,'like',R);
            rmin=min(R,[],1); rmax=max(R,[],1);
        end
        Rc=R-rc;
        rhoBlock=phase_radius_box_exp3(qModes,rmin,rmax,rc);
        rhoMax=max(rhoMax,rhoBlock);

        phaseCenter=exp(1i*(qModes*rc(:)));
        cCenter=coeff.*phaseCenter;

        mPts=size(Rc,1);
        Rx=ones(mPts,p+1,'like',Rc); Ry=Rx; Rz=Rx;
        for s=1:p
            Rx(:,s+1)=Rx(:,s).*Rc(:,1);
            Ry(:,s+1)=Ry(:,s).*Rc(:,2);
            Rz(:,s+1)=Rz(:,s).*Rc(:,3);
        end

        ub=complex(zeros(mPts,1,'like',coeff));
        for s0=1:termBatchSize:L
            s1=min(L,s0+termBatchSize-1);
            aa=alphaList(s0:s1,:);
            a1=aa(:,1).'; a2=aa(:,2).'; a3=aa(:,3).';
            Qb=Qx(:,a1+1).*Qy(:,a2+1).*Qz(:,a3+1);
            Rb=Rx(:,a1+1).*Ry(:,a2+1).*Rz(:,a3+1);
            mom=Qb.'*cCenter;
            mom=coeffAlpha(s0:s1).*mom;
            ub=ub+Rb*mom;
        end
        u(ids)=ub;

        maxBlockPts=max(maxBlockPts,mPts);
        maxCenterAux=max(maxCenterAux,estimate_center_aux_bytes(coeff,NF,mPts,p,termBatchSize));
    end
    wait_for_gpu(u);
    t=toc(t0);

    bytesReal=bytes_per_real(coeff);
    qPowBytes=3*double(NF)*double(p+1)*bytesReal;
    alphaBytes=double(L)*2*bytesReal; % complex Taylor coefficients
    peakBytes=qPowBytes+alphaBytes+maxCenterAux;

    info=struct();
    info.order=p;
    info.numTerms=L;
    info.termBatchSize=termBatchSize;
    info.numCenters=numel(blocks);
    info.rhoMax=rhoMax;
    info.timeSeconds=t;
    info.estimatedPeakAuxBytes=peakBytes;
    info.maxBlockPoints=maxBlockPts;
end

function rho=phase_radius_box_exp3(qModes,rmin,rmax,rc)
    % Exact maximum over the rectangular point hull, attained at a corner.
    corners=zeros(8,3,'like',qModes);
    c=0;
    for a=0:1
        for b=0:1
            for d=0:1
                c=c+1;
                corners(c,1)=rmin(1)+(rmax(1)-rmin(1))*a;
                corners(c,2)=rmin(2)+(rmax(2)-rmin(2))*b;
                corners(c,3)=rmin(3)+(rmax(3)-rmin(3))*d;
            end
        end
    end
    D=corners-rc;
    qd=qModes*D.';
    rho=gather_scalar(max(abs(qd(:))));
end

function bytes=estimate_center_aux_bytes(coeff,NF,mPts,p,batchTerms)
    br=bytes_per_real(coeff); bc=2*br;
    b=min(batchTerms,nchoosek(p+3,3));
    rPow=3*double(mPts)*double(p+1)*br;
    phaseCenter=double(NF)*bc;
    cCenter=double(NF)*bc;
    Qb=double(NF)*double(b)*br;
    Rb=double(mPts)*double(b)*br;
    moments=double(b)*bc;
    ub=double(mPts)*bc;
    Rcopy=double(mPts)*3*br;
    bytes=rPow+phaseCenter+cCenter+Qb+Rb+moments+ub+Rcopy;
end

function b=bytes_per_real(x)
    if isa(x,'gpuArray'), c=classUnderlying(x); else, c=class(x); end
    if strcmp(c,'single'), b=4; else, b=8; end
end
