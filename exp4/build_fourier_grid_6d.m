function fourier = build_fourier_grid_6d(N,dim,T,prototype)
%BUILD_FOURIER_GRID_6D Build tensor Fourier frequencies in the old ordering.

    if dim ~= 6
        error('The current Maxwell embedding code expects dim=6.');
    end
    if ~isa(prototype,'gpuArray')
        error('Fourier grid construction requires a gpuArray prototype.');
    end

    nGrid = 2*N;
    sz = nGrid*ones(1,dim);
    n = prod(sz);

    freq1CPU = [0:N-1, -N:-1];
    freq1 = cast(freq1CPU,'like',prototype);
    [J1,J2,J3,J4,J5,J6] = ndgrid(freq1,freq1,freq1,freq1,freq1,freq1);

    Xi = cell(dim,1);
    Xi{1}=(2*pi/T(1))*J1; Xi{2}=(2*pi/T(2))*J2;
    Xi{3}=(2*pi/T(3))*J3; Xi{4}=(2*pi/T(4))*J4;
    Xi{5}=(2*pi/T(5))*J5; Xi{6}=(2*pi/T(6))*J6;

    fourier = struct();
    fourier.N=N;
    fourier.dim=dim;
    fourier.nGrid=nGrid;
    fourier.sz=sz;
    fourier.n=n;
    fourier.ndof=3*n;
    fourier.freq1=freq1CPU;
    fourier.Xi=Xi;
end
