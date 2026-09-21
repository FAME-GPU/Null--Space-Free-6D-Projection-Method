function [Kop,U0Hop,D0_vec,Vdata,sigma_r] = ...
    assemble_K_and_nullspace_free_basis(P,k,Xi,sz,n,d)
%ASSEMBLE_K_AND_NULLSPACE_FREE_BASIS GPU matrix-free spectral operators.
%
% This GPU version avoids assembling the sparse 3n-by-3n curl-curl matrix K,
% the sparse null-space basis U0, and the sparse 3n-by-2n transverse basis Ur.
% Instead it returns:
%
%   Kop(x)   = K*x, applied mode by mode using diagonal wave-number arrays;
%   U0Hop(x) = U0^* x, applied mode by mode;
%   D0_vec   = |q|;
%   Vdata    = six diagonals of V = Ur*Sigma^(1/2);
%   sigma_r  = [|q|^2; |q|^2].
%
% All large arrays remain on the same execution device and in the same
% precision as Xi{1}.

    q1 = zeros(sz,'like',Xi{1});
    q2 = zeros(sz,'like',Xi{1});
    q3 = zeros(sz,'like',Xi{1});

    for ell = 1:d
        shiftedXi = Xi{ell} + k(ell);
        q1 = q1 + P(ell,1) .* shiftedXi;
        q2 = q2 + P(ell,2) .* shiftedXi;
        q3 = q3 + P(ell,3) .* shiftedXi;
    end

    q1v = q1(:);
    q2v = q2(:);
    q3v = q3(:);

    Qq = abs(q1v).^2 + abs(q2v).^2 + abs(q3v).^2;
    sqrtQq = sqrt(Qq);

    tol = 1e-13;
    minQq = gather_scalar(min(Qq));
    if minQq <= tol
        error(['A zero or nearly zero projected Fourier mode was detected. ', ...
               'The current null-space-free reduction requires Gamma-point ', ...
               'regularization so that every |q_xi| is nonzero. min |q|^2 = %.3e'],minQq);
    end

    Qs = q1v + q2v + q3v;
    denom = 3*Qq - abs(Qs).^2;

    regularMask = denom > tol;
    singularMask = ~regularMask;

    % Regular transverse basis, evaluated vectorwise on the GPU.
    denomSafe=max(denom,cast(tol,'like',denom));
    scale1=1./sqrt(Qq.*denomSafe);
    u11=(Qq-q1v.*conj(Qs)).*scale1;
    u21=(Qq-q2v.*conj(Qs)).*scale1;
    u31=(Qq-q3v.*conj(Qs)).*scale1;
    scale2=1./sqrt(denomSafe);
    u12=(conj(q3v)-conj(q2v)).*scale2;
    u22=(conj(q1v)-conj(q3v)).*scale2;
    u32=(conj(q2v)-conj(q1v)).*scale2;

    % Rare singular transverse modes are also handled entirely on GPU.
    if gather_scalar(any(singularMask))
        qh1=q1v./sqrtQq; qh2=q2v./sqrtQq; qh3=q3v./sqrtQq;
        [~,imin]=min([abs(qh1),abs(qh2),abs(qh3)],[],2);
        a1=cast(imin==1,'like',qh1); a2=cast(imin==2,'like',qh1); a3=cast(imin==3,'like',qh1);
        dotqa=conj(qh1).*a1+conj(qh2).*a2+conj(qh3).*a3;
        z11=a1-qh1.*dotqa; z21=a2-qh2.*dotqa; z31=a3-qh3.*dotqa;
        zn=sqrt(abs(z11).^2+abs(z21).^2+abs(z31).^2);
        z11=z11./zn; z21=z21./zn; z31=z31./zn;
        z12=qh2.*z31-qh3.*z21;
        z22=qh3.*z11-qh1.*z31;
        z32=qh1.*z21-qh2.*z11;
        z2n=sqrt(abs(z12).^2+abs(z22).^2+abs(z32).^2);
        z12=z12./z2n; z22=z22./z2n; z32=z32./z2n;
        sm=cast(singularMask,'like',q1v); rm=1-sm;
        u11=rm.*u11+sm.*z11; u21=rm.*u21+sm.*z21; u31=rm.*u31+sm.*z31;
        u12=rm.*u12+sm.*z12; u22=rm.*u22+sm.*z22; u32=rm.*u32+sm.*z32;
        clear qh1 qh2 qh3 imin a1 a2 a3 dotqa z11 z21 z31 zn z12 z22 z32 z2n sm rm
    end

    sigma_r = [Qq;Qq];
    D0_vec = sqrtQq;

    % Store V=Ur*Sigma^(1/2) directly by its six modewise diagonals.
    Vdata = struct();
    Vdata.V11 = u11 .* sqrtQq;
    Vdata.V21 = u21 .* sqrtQq;
    Vdata.V31 = u31 .* sqrtQq;
    Vdata.V12 = u12 .* sqrtQq;
    Vdata.V22 = u22 .* sqrtQq;
    Vdata.V32 = u32 .* sqrtQq;
    Vdata.n = n;
    Vdata.nr = 2*n;

    Kop = @applyK;
    U0Hop = @applyU0H;

    function y = applyK(x)
        if size(x,1) ~= 3*n
            error('Kop input must have 3*n rows.');
        end
        x1 = x(1:n,:); x2 = x(n+1:2*n,:); x3 = x(2*n+1:3*n,:);
        y1 = (abs(q2v).^2+abs(q3v).^2).*x1 - (q1v.*conj(q2v)).*x2 - (q1v.*conj(q3v)).*x3;
        y2 = -(q2v.*conj(q1v)).*x1 + (abs(q1v).^2+abs(q3v).^2).*x2 - (q2v.*conj(q3v)).*x3;
        y3 = -(q3v.*conj(q1v)).*x1 - (q3v.*conj(q2v)).*x2 + (abs(q1v).^2+abs(q2v).^2).*x3;
        y = [y1;y2;y3];
    end

    function y = applyU0H(x)
        if size(x,1) ~= 3*n
            error('U0Hop input must have 3*n rows.');
        end
        x1 = x(1:n,:); x2 = x(n+1:2*n,:); x3 = x(2*n+1:3*n,:);
        y = conj(q1v./sqrtQq).*x1 + conj(q2v./sqrtQq).*x2 + conj(q3v./sqrtQq).*x3;
    end
end
