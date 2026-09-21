function sys = build_experiment1_q_system_A_only(common,qPhysCPU,cfg)
%BUILD_EXPERIMENT1_Q_SYSTEM_A_ONLY Build only the original K/M operators.
%
% This deliberately avoids constructing the null-space-free transverse basis
% and all Method-B/C objects.

    P = common.P;
    n = common.NF;
    d = cfg.problem.dim;
    Xi = common.fourier.Xi;
    sz = common.fourier.sz;

    qPhys = to_gpu(qPhysCPU(:),cfg.gpu.precision);
    kLift = P*((P.'*P)\qPhys);

    q1 = zeros(sz,'like',Xi{1});
    q2 = zeros(sz,'like',Xi{1});
    q3 = zeros(sz,'like',Xi{1});

    for ell = 1:d
        shiftedXi = Xi{ell} + kLift(ell);
        q1 = q1 + P(ell,1).*shiftedXi;
        q2 = q2 + P(ell,2).*shiftedXi;
        q3 = q3 + P(ell,3).*shiftedXi;
    end

    q1v = q1(:);
    q2v = q2(:);
    q3v = q3(:);
    clear q1 q2 q3

    Qq = abs(q1v).^2 + abs(q2v).^2 + abs(q3v).^2;
    sqrtQq = sqrt(Qq);
    minQq = gather_scalar(min(Qq));

    if minQq <= 1e-13
        error(['A zero or nearly zero projected Fourier mode was detected. ', ...
               'This one-point test assumes q_xi is nonzero for all retained modes. ', ...
               'min |q_xi|^2 = %.3e'],minQq);
    end

    sys = struct();
    sys.N = common.N;
    sys.NF = n;
    sys.ndof = 3*n;
    sys.P = P;
    sys.qPhys = qPhys;
    sys.kLift = kLift;
    sys.mass = common.mass;
    sys.Kop = @applyK;
    sys.U0Hop = @applyU0H;
    sys.minQq = minQq;
    % Expose the projected Fourier wave vectors so the spectral
    % preconditioner can reuse them without rebuilding the grid.
    sys.q1v = q1v;
    sys.q2v = q2v;
    sys.q3v = q3v;

    function y = applyK(x)
        if size(x,1) ~= 3*n
            error('Kop input must have 3*NF rows.');
        end
        x1 = x(1:n,:);
        x2 = x(n+1:2*n,:);
        x3 = x(2*n+1:3*n,:);

        y1 = (abs(q2v).^2+abs(q3v).^2).*x1 ...
            - (q1v.*conj(q2v)).*x2 - (q1v.*conj(q3v)).*x3;
        y2 = -(q2v.*conj(q1v)).*x1 ...
            + (abs(q1v).^2+abs(q3v).^2).*x2 - (q2v.*conj(q3v)).*x3;
        y3 = -(q3v.*conj(q1v)).*x1 - (q3v.*conj(q2v)).*x2 ...
            + (abs(q1v).^2+abs(q2v).^2).*x3;
        y = [y1;y2;y3];
    end

    function y = applyU0H(x)
        if size(x,1) ~= 3*n
            error('U0Hop input must have 3*NF rows.');
        end
        x1 = x(1:n,:);
        x2 = x(n+1:2*n,:);
        x3 = x(2*n+1:3*n,:);
        y = conj(q1v./sqrtQq).*x1 ...
          + conj(q2v./sqrtQq).*x2 ...
          + conj(q3v./sqrtQq).*x3;
    end
end
