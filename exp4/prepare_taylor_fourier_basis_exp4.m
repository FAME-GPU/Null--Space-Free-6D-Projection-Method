function basis = prepare_taylor_fourier_basis_exp4(qModes,p)
%PREPARE_TAYLOR_FOURIER_BASIS_EXP4 Precompute all q^alpha, |alpha|<=p.
%
% This is the memory/performance key for the N=5 Exp4 reconstruction.
% Qalpha is independent of the Yee mesh, window, center, field component,
% and tracked mode, so it is built only once and reused throughout Exp4.

    if ~isa(qModes,'gpuArray')
        error('Taylor Fourier basis must be constructed on the GPU.');
    end
    NF=size(qModes,1);
    alphaList=build_multi_index_list_3d_exp3(p);
    L=size(alphaList,1);

    Qx=ones(NF,p+1,'like',qModes);
    Qy=Qx; Qz=Qx;
    for s=1:p
        Qx(:,s+1)=Qx(:,s).*qModes(:,1);
        Qy(:,s+1)=Qy(:,s).*qModes(:,2);
        Qz(:,s+1)=Qz(:,s).*qModes(:,3);
    end

    Qalpha=zeros(NF,L,'like',qModes);
    for ia=1:L
        a=alphaList(ia,:);
        Qalpha(:,ia)=Qx(:,a(1)+1).*Qy(:,a(2)+1).*Qz(:,a(3)+1);
    end
    clear Qx Qy Qz

    coeffAlphaCPU=complex(zeros(L,1));
    for ia=1:L
        a=alphaList(ia,:);
        coeffAlphaCPU(ia)=(1i)^sum(a)*exp(-sum(gammaln(a+1)));
    end
    coeffAlpha=gpuArray(cast(coeffAlphaCPU,classUnderlying(qModes)));

    qL1Max=gather_scalar(max(sum(abs(qModes),2)));
    basis=struct();
    basis.p=p;
    basis.alphaList=alphaList;
    basis.numTerms=L;
    basis.Qalpha=Qalpha;
    basis.coeffAlpha=coeffAlpha;
    basis.qModes=qModes;
    basis.qL1Max=qL1Max;
    basis.estimatedQalphaBytes=double(NF)*double(L)*bytes_per_real(qModes);
end

function b=bytes_per_real(x)
    if isa(x,'gpuArray'), c=classUnderlying(x); else, c=class(x); end
    if strcmp(c,'single'), b=4; else, b=8; end
end
