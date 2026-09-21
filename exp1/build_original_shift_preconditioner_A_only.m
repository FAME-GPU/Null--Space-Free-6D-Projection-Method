function [Pinv,meta] = build_original_shift_preconditioner_A_only(sys,tau)
%BUILD_ORIGINAL_SHIFT_PRECONDITIONER_A_ONLY Spectral SPD preconditioner.
%
%   P_A = K + tau*I, tau>0.
%
% For every retained Fourier mode, K_q = |q|^2 I - q q^*.  Therefore
%
%   (K_q+tau I)^(-1)
%     = 1/(Q+tau) I
%       + (1/tau-1/(Q+tau)) (q q^*)/Q,
%
% where Q=|q|^2.  This inverse is applied exactly mode by mode on the GPU.
% No sparse/dense factorization is formed.

    if tau <= 0 || ~isfinite(tau)
        error('Preconditioner tau must be finite and positive.');
    end
    n = sys.NF;
    q1 = sys.q1v;
    q2 = sys.q2v;
    q3 = sys.q3v;

    Q = abs(q1).^2 + abs(q2).^2 + abs(q3).^2;
    minQ = gather_scalar(min(Q));
    maxQ = gather_scalar(max(Q));
    if minQ <= 0
        error('Zero projected Fourier wave vector in preconditioner construction.');
    end

    tauLike = cast(tau,'like',Q);
    invTrans = 1./(Q+tauLike);
    corr = (1./tauLike-invTrans)./Q;
    clear Q tauLike

    Pinv = @applyPinv;
    meta = struct( ...
        'name','spectral-(K+tauI)', ...
        'tau',tau, ...
        'minQ',minQ, ...
        'maxQ',maxQ, ...
        'factorization','none; exact modewise Fourier-symbol inverse');

    function y = applyPinv(x)
        if size(x,1) ~= 3*n
            error('Preconditioner input must have 3*NF rows.');
        end
        x1 = x(1:n,:);
        x2 = x(n+1:2*n,:);
        x3 = x(2*n+1:3*n,:);
        qHx = conj(q1).*x1 + conj(q2).*x2 + conj(q3).*x3;
        cqx = corr.*qHx;
        y1 = invTrans.*x1 + q1.*cqx;
        y2 = invTrans.*x2 + q2.*cqx;
        y3 = invTrans.*x3 + q3.*cqx;
        y = [y1;y2;y3];
    end
end
