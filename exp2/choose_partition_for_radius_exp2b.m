function out = choose_partition_for_radius_exp2b(n,h,qL1Max,rhoTarget,relTol)
%CHOOSE_PARTITION_FOR_RADIUS_EXP2B Smallest k with r_ph,max <= target.
%
% For the nearly equal k-by-k-by-k Cartesian partition used here, the
% largest side length (in Yee index points) is ceil((2*n+3)/k).  Because
% each local center is the midpoint of its block, the exact global phase
% radius is
%
%   r_ph,max = 0.5*h*(Lmax-1)*max_j ||q_j||_1.
%
% This is exactly the corner maximum used by the original implementation,
% but avoids constructing any Yee coordinates merely to choose k.

    arguments
        n (1,1) double {mustBeInteger,mustBeNonnegative}
        h (1,1) double {mustBePositive}
        qL1Max (1,1) double {mustBeNonnegative}
        rhoTarget (1,1) double {mustBeNonnegative}
        relTol (1,1) double {mustBeNonnegative} = 1e-12
    end

    m = 2*n + 3;
    if qL1Max == 0
        k = 1;
        rho = 0;
    else
        k = 1;
        limit = rhoTarget*(1+relTol);
        while true
            Lmax = ceil(m/k);
            rho = 0.5*h*(Lmax-1)*qL1Max;
            if rho <= limit
                break;
            end
            k = k + 1;
            if k > m
                error('choose_partition_for_radius_exp2b:NoFeasiblePartition', ...
                    'Failed to find a partition satisfying the phase-radius target.');
            end
        end
    end

    out = struct();
    out.n = n;
    out.numPointsPerDimension = m;
    out.blocksPerDimension = k;
    out.numCenters = k^3;
    out.maxBlockSidePoints = ceil(m/k);
    out.rhoMax = rho;
    out.rhoTarget = rhoTarget;
end
