function pre = precompute_taylor_exp2b(qModes,p,termBatchSize)
%PRECOMPUTE_TAYLOR_EXP2B Fourier-side data reused by all N_G values.
%
% This preprocessing is deliberately outside the reconstruction timers,
% matching the existing Experiment 2 timing convention: q powers depend on
% N_F and p, not on the Yee-grid size N_G.

    if ~isa(qModes,'gpuArray')
        error('precompute_taylor_exp2b:GPURequired','qModes must be a gpuArray.');
    end
    NF = size(qModes,1);
    if size(qModes,2) ~= 3
        error('precompute_taylor_exp2b:BadQ','qModes must be NF-by-3.');
    end
    p = round(p);
    if p < 0
        error('precompute_taylor_exp2b:BadOrder','Taylor order must be nonnegative.');
    end

    alphaList = build_multi_index_list_3d_exp2b(p);
    L = size(alphaList,1);
    termBatchSize = max(1,min(L,round(termBatchSize)));

    Qx = ones(NF,p+1,'like',qModes);
    Qy = ones(NF,p+1,'like',qModes);
    Qz = ones(NF,p+1,'like',qModes);
    for s = 1:p
        Qx(:,s+1) = Qx(:,s).*qModes(:,1);
        Qy(:,s+1) = Qy(:,s).*qModes(:,2);
        Qz(:,s+1) = Qz(:,s).*qModes(:,3);
    end

    coeffAlphaCPU = zeros(L,1);
    for ia = 1:L
        a = alphaList(ia,:);
        coeffAlphaCPU(ia) = (1i)^sum(a)*exp(-sum(gammaln(a+1)));
    end
    % Make complex coefficients on the same device/precision as qModes.
    coeffAlphaR = cast(real(coeffAlphaCPU),'like',qModes(1));
    coeffAlphaI = cast(imag(coeffAlphaCPU),'like',qModes(1));
    coeffAlpha = complex(coeffAlphaR,coeffAlphaI);

    qL1Max = gather_scalar(max(sum(abs(qModes),2)));

    pre = struct();
    pre.p = p;
    pre.alphaList = alphaList;
    pre.numTerms = L;
    pre.termBatchSize = termBatchSize;
    pre.Qx = Qx;
    pre.Qy = Qy;
    pre.Qz = Qz;
    pre.coeffAlpha = coeffAlpha;
    pre.qL1Max = qL1Max;
end
