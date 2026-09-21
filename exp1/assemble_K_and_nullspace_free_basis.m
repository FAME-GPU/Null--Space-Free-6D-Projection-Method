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

    % Ur components. The first subscript is the physical component and the
    % second is the transverse family. They are initialized on the GPU.
    u11 = zeros(n,1,'like',q1v); u21 = zeros(n,1,'like',q1v); u31 = zeros(n,1,'like',q1v);
    u12 = zeros(n,1,'like',q1v); u22 = zeros(n,1,'like',q1v); u32 = zeros(n,1,'like',q1v);

    idxReg = find(regularMask);
    if ~isempty(idxReg)
        q1r = q1v(idxReg); q2r = q2v(idxReg); q3r = q3v(idxReg);
        Qqr = Qq(idxReg); Qsr = Qs(idxReg); denomr = denom(idxReg);

        scale1 = 1 ./ sqrt(Qqr .* denomr);
        u11(idxReg) = (Qqr - q1r .* conj(Qsr)) .* scale1;
        u21(idxReg) = (Qqr - q2r .* conj(Qsr)) .* scale1;
        u31(idxReg) = (Qqr - q3r .* conj(Qsr)) .* scale1;

        scale2 = 1 ./ sqrt(denomr);
        u12(idxReg) = (conj(q3r) - conj(q2r)) .* scale2;
        u22(idxReg) = (conj(q1r) - conj(q3r)) .* scale2;
        u32(idxReg) = (conj(q2r) - conj(q1r)) .* scale2;
    end

    % The singular branch is normally empty for the irrational projection
    % used in the supplied experiment. It is handled on the CPU only for the
    % few 3-vectors involved; the completed basis vectors are copied back to
    % the GPU. This avoids any sparse GPU construction.
    idxSing = gather(find(singularMask));
    for ss = 1:numel(idxSing)
        m = idxSing(ss);
        q = gather([q1v(m);q2v(m);q3v(m)]);
        qnorm = norm(q);
        qhat = q/qnorm;
        [~,imin] = min(abs(qhat));
        a = zeros(3,1);
        a(imin) = 1;
        z1 = a - qhat*(qhat'*a);
        z1 = z1/norm(z1);
        z2 = cross(qhat,z1);
        z2 = z2/norm(z2);

        u11(m) = z1(1); u21(m) = z1(2); u31(m) = z1(3);
        u12(m) = z2(1); u22(m) = z2(2); u32(m) = z2(3);
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
