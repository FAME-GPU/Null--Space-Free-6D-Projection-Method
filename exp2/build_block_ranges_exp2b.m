function info = build_block_ranges_exp2b(n,k)
%BUILD_BLOCK_RANGES_EXP2B Compact Cartesian block metadata only.
%
% Returns one row [i0 i1 j0 j1 k0 k1] per local block.  Unlike the old
% partition builder, this function never stores a length-N_G vector of
% linear indices, so metadata remains small even at n=200.

    arguments
        n (1,1) double {mustBeInteger,mustBeNonnegative}
        k (1,1) double {mustBeInteger,mustBePositive}
    end

    m = 2*n + 3;
    if k > m
        error('build_block_ranges_exp2b:TooManyBlocks', ...
            'blocksPerDimension=%d exceeds points per dimension=%d.',k,m);
    end

    idxMin = -(n+1);
    lengths = floor(m/k)*ones(k,1);
    lengths(1:mod(m,k)) = lengths(1:mod(m,k)) + 1;

    starts = zeros(k,1);
    ends = zeros(k,1);
    s = idxMin;
    for b = 1:k
        starts(b) = s;
        ends(b) = s + lengths(b) - 1;
        s = ends(b) + 1;
    end
    if s ~= n+2
        error('build_block_ranges_exp2b:CoverageBug','1-D partition does not cover the plus grid exactly.');
    end

    nCenters = k^3;
    ranges = zeros(nCenters,6,'int32');
    c = 0;
    maxPts = 0;
    minPts = inf;
    for bz = 1:k
        for by = 1:k
            for bx = 1:k
                c = c + 1;
                ranges(c,:) = int32([starts(bx) ends(bx) starts(by) ends(by) starts(bz) ends(bz)]);
                np = lengths(bx)*lengths(by)*lengths(bz);
                maxPts = max(maxPts,np);
                minPts = min(minPts,np);
            end
        end
    end

    if c ~= nCenters
        error('build_block_ranges_exp2b:CountBug','Unexpected number of local blocks.');
    end
    totalPts = sum(double(ranges(:,2)-ranges(:,1)+1) .* ...
                   double(ranges(:,4)-ranges(:,3)+1) .* ...
                   double(ranges(:,6)-ranges(:,5)+1));
    if totalPts ~= m^3
        error('build_block_ranges_exp2b:CoverageBug', ...
            '3-D block coverage is %g points but expected %g.',totalPts,m^3);
    end

    info = struct();
    info.ranges = ranges;
    info.numCenters = nCenters;
    info.blocksPerDimension = k;
    info.numPointsPerDimension = m;
    info.numPoints = m^3;
    info.maxBlockPoints = maxPts;
    info.minBlockPoints = minPts;
    info.lengths1D = lengths;
end
