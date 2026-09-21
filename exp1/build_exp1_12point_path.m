function path = build_exp1_12point_path(cfg)
%BUILD_EXP1_12POINT_PATH Four vertices + two interior points per segment.
%
% Independent points are ordered as
%   q0, 1/3(q0->q1), 2/3(q0->q1), q1, ...,
%   q3, 1/3(q3->q0), 2/3(q3->q0).
% The repeated closing q0 is added only by the local plotting code.

    V=cfg.path.vertices;
    if ~isequal(size(V),[3 5]) || norm(V(:,1)-V(:,5))>10*eps
        error('cfg.path.vertices must be 3x5 with the fifth vertex equal to the first.');
    end
    f=cfg.path.interiorFractions(:).';
    if numel(f)~=2 || any(abs(f-[1/3 2/3])>1e-14)
        error('Formal Experiment 1 uses interior fractions [1/3 2/3].');
    end

    q=zeros(3,12);
    segment=zeros(12,1);
    localFraction=zeros(12,1);
    pathCoordinate=zeros(12,1);
    kk=0;
    for s=1:4
        vals=[0 f];
        for j=1:3
            kk=kk+1;
            t=vals(j);
            q(:,kk)=(1-t)*V(:,s)+t*V(:,s+1);
            segment(kk)=s;
            localFraction(kk)=t;
            pathCoordinate(kk)=(s-1)+t;
        end
    end

    path=struct();
    path.q=q;
    path.segment=segment;
    path.localFraction=localFraction;
    path.pathCoordinate=pathCoordinate;
    path.numIndependent=12;
    path.qPlot=[q q(:,1)];
    path.pathCoordinatePlot=[pathCoordinate;4];
    path.endpointPlotIndices=[1 4 7 10 13];
    path.endpointLabels={'q^{(0)}','q^{(1)}','q^{(2)}','q^{(3)}','q^{(0)}'};
end
