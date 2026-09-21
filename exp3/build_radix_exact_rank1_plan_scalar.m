function plan = build_radix_exact_rank1_plan_scalar(N,dim,precision,useGPU)
%BUILD_RADIX_EXACT_RANK1_PLAN_SCALAR Exact radix rank-1 plan for tensor cubes.
%
% For nGrid=2*N and B=2*nGrid-1=4*N-1, choose
%   z = [1,B,...,B^(dim-1)]^T,
%   M = nGrid*B^(dim-1).
% The full shifted difference code is injective in the first dim-1 radix-B
% digits and the last digit is reduced modulo nGrid.  The only collisions
% differ by nGrid in the last shifted difference digit; these representatives
% have the same tensor circular residue.  Hence the plan is universally exact.
%
% Unlike the generic verifier, this constructor does not enumerate and sort
% the complete B^dim difference box.  This is essential for N=5, dim=6.

    if nargin<3 || isempty(precision), precision='double'; end
    if nargin<4 || isempty(useGPU), useGPU=true; end

    nGrid = 2*N;
    B = 2*nGrid-1;
    NF = nGrid^dim;
    z = double(B).^(0:dim-1).';
    M = double(nGrid)*double(B)^(dim-1);
    s = nGrid-1;

    if M >= double(intmax('uint32'))
        error('Rank-1 length M=%g exceeds uint32 indexing range.',M);
    end

    % Retained tensor inputs k_j=0,...,nGrid-1 have distinct base-B codes.
    idx0=(0:NF-1).';
    tmp=idx0;
    inputCode=zeros(NF,1);
    for j=1:dim
        kj=mod(tmp,nGrid);
        tmp=floor(tmp/nGrid);
        inputCode=inputCode+kj*z(j);
    end
    if max(inputCode)>=M
        error('Unexpected retained radix code outside [0,M-1].');
    end

    shiftCode=mod(s*sum(z),M);
    targetCode=mod(inputCode+shiftCode,M);

    plan=struct();
    plan.method='universally exact radix rank-1 convolution (scalar optimized)';
    plan.N=N;
    plan.dim=dim;
    plan.nGrid=nGrid;
    plan.sz=nGrid*ones(1,dim);
    plan.NF=NF;
    plan.B=B;
    plan.M=M;
    plan.z=z;
    plan.embeddingRatio=M/NF;
    plan.shiftCode=shiftCode;
    plan.inputIdx=uint32(inputCode+1);
    plan.targetIdx=uint32(targetCode+1);
    plan.precision=precision;
    plan.useGPU=logical(useGPU);
    plan.exactCertified=true;
    plan.kernelIsFull=true;
    plan.kernelChunkSize=1e6;
end
