function [field,info] = reconstruct_single_mode_large_window_exp3(Xhost,basis,n,h,part,cfg)
%RECONSTRUCT_SINGLE_MODE_LARGE_WINDOW_EXP3 Memory-efficient L=4 reconstruction.
%
% The large window needs many Taylor centers (30^3 for the present setup).
% This implementation therefore avoids both extremes that are undesirable at
% n=160: it does not store global M-by-(p+1) Rx/Ry/Rz arrays, and it does not
% launch one polynomial evaluation per 3D block.  Instead it stores only a
% one-dimensional local-power table on the regular grid.  For each Taylor
% multi-index, the center moment is expanded piecewise-constantly from the
% 30^3 center grid to the 323^3 Yee grid and multiplied by the three 1D
% local-power factors.  The arithmetic matches the original multi-center
% Taylor formula while keeping GPU memory bounded.

    assert_gpu_basis_alive_exp3(basis,'Large-window reconstruction (entry)');

    NF = size(basis.qModes,1);
    if numel(Xhost) ~= 3*NF
        error('Expected one Fourier eigenvector with 3*NF entries.');
    end
    if part.n ~= n || abs(part.h-h)>1e-14
        error('Large-window partition metadata does not match n/h.');
    end

    p = cfg.taylor.order;
    p1 = p+1;
    L = basis.numTerms;
    nb = part.blocksPerDimension;
    C = part.numCenters;
    m = 2*n+3;
    groups = part.groups;
    batch = max(1,min(C,round(cfg.taylor.largeWindowCenterBatchSize)));

    if C ~= nb^3 || numel(groups) ~= nb
        error('Invalid lightweight Taylor partition.');
    end

    % 1D block ID and local coordinates.  The staggered half-cell shifts
    % cancel after centering, so this same local geometry serves u1,u2,u3.
    blockId1D = zeros(m,1);
    local1D = zeros(m,1);
    for ib = 1:nb
        g = groups{ib};
        idx = double(g) - double(n+2); % -(n+1):n+1
        xc = (idx(1)+idx(end))/2;
        blockId1D(g) = ib;
        local1D(g) = h*(idx(:)-xc);
    end
    if any(blockId1D==0)
        error('Lightweight partition does not cover the full 1D grid.');
    end
    blockId1D = double(blockId1D);
    xLocal = gpuArray(cast(local1D,classUnderlying(basis.qModes)));
    P1D = ones(m,p1,'like',basis.qModes);
    for a = 1:p
        P1D(:,a+1) = P1D(:,a).*xLocal;
    end
    clear xLocal local1D

    shifts = [0.5 0 0; 0 0.5 0; 0 0 0.5];
    U = cell(3,1);
    componentInfo = cell(3,1);
    totalTimer = tic;

    for ell = 1:3
        rows = (ell-1)*NF + (1:NF);
        coeff = gpuArray(cast(Xhost(rows),classUnderlying(basis.qModes)));
        coeff = coeff(:);

        centersCPU = build_component_centers(groups,n,h,shifts(ell,:));
        moments = complex(zeros(L,C,'like',coeff));

        wait_for_gpu();
        tMoment = tic;
        for c0 = 1:batch:C
            c1 = min(C,c0+batch-1);
            ids = c0:c1;
            centers = gpuArray(cast(centersCPU(ids,:),classUnderlying(basis.qModes)));
            phase = exp(1i*(basis.qModes*centers.'));
            cStack = phase.*coeff;
            mom = basis.Qalpha.'*cStack;
            mom = basis.coeffAlpha.*mom;
            moments(:,ids) = mom;
            clear centers phase cStack mom
        end
        wait_for_gpu(moments);
        momentTime = toc(tMoment);

        u = complex(zeros(m,m,m,'like',coeff));
        wait_for_gpu();
        tEval = tic;
        for ia = 1:L
            a = basis.alphaList(ia,:);
            centerGrid = reshape(moments(ia,:),[nb nb nb]);
            momentFull = centerGrid(blockId1D,blockId1D,blockId1D);
            px = reshape(P1D(:,a(1)+1),[m 1 1]);
            py = reshape(P1D(:,a(2)+1),[1 m 1]);
            pz = reshape(P1D(:,a(3)+1),[1 1 m]);
            u = u + momentFull.*px.*py.*pz;
            clear centerGrid momentFull px py pz
        end
        wait_for_gpu(u);
        evalTime = toc(tEval);

        U{ell} = u;
        componentInfo{ell} = struct('momentTimeSeconds',momentTime, ...
            'evaluationTimeSeconds',evalTime,'totalTimeSeconds',momentTime+evalTime, ...
            'numCenters',C,'centerBatchSize',batch,'numTerms',L);
        clear coeff moments centersCPU u
    end

    wait_for_gpu(U{3});
    totalTime = toc(totalTimer);
    field = struct('u1',U{1},'u2',U{2},'u3',U{3});
    info = struct('timeSeconds',totalTime,'componentInfo',{componentInfo}, ...
        'numCenters',C,'blocksPerDimension',nb,'rhoBound',part.rhoBound, ...
        'numTerms',L,'method','streaming-large-window-1D-geometry');
    clear U P1D blockId1D
end

function centers = build_component_centers(groups,n,h,shift)
    nb = numel(groups);
    centers = zeros(nb^3,3);
    c = 0;
    for kb = 1:nb
        kz = center_index(groups{kb},n) + shift(3);
        for jb = 1:nb
            jy = center_index(groups{jb},n) + shift(2);
            for ib = 1:nb
                ix = center_index(groups{ib},n) + shift(1);
                c = c+1;
                centers(c,:) = h*[ix jy kz];
            end
        end
    end
end

function x = center_index(g,n)
    left = double(g(1)) - double(n+2);
    right = double(g(end)) - double(n+2);
    x = (left+right)/2;
end
