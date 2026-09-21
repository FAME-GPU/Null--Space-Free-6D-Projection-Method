function [u,info] = reconstruct_field_direct_blocked_exp3(coeff,qModes,rPoints,blockSize)
%RECONSTRUCT_FIELD_DIRECT_BLOCKED_EXP3 Direct Fourier sum in point batches.
    coeff=coeff(:);
    M=size(rPoints,1); NF=size(qModes,1);
    if numel(coeff)~=NF, error('Coefficient length does not match qModes.'); end
    if nargin<4 || isempty(blockSize), blockSize=512; end

    u=complex(zeros(M,1,'like',coeff));
    wait_for_gpu(); t0=tic;
    for s=1:blockSize:M
        e=min(M,s+blockSize-1);
        ids=s:e;
        phase=rPoints(ids,:)*qModes.';
        u(ids)=exp(1i*phase)*coeff;
    end
    wait_for_gpu(u);
    t=toc(t0);

    % Auxiliary memory estimate excludes coeff and final u.  At the peak,
    % the real phase block and its complex exponential coexist.
    bytesReal=bytes_per_real(coeff);
    bytesComplex=2*bytesReal;
    b=min(blockSize,M);
    peakBytes=(bytesReal+bytesComplex)*double(b)*double(NF);

    info=struct('method','blockwise direct','timeSeconds',t, ...
        'blockSize',blockSize,'estimatedPeakAuxBytes',peakBytes, ...
        'numPoints',M,'numModes',NF);
end

function b=bytes_per_real(x)
    if isa(x,'gpuArray'), c=classUnderlying(x); else, c=class(x); end
    if strcmp(c,'single'), b=4; else, b=8; end
end
