function path = build_exp4_path(cfg)
%BUILD_EXP4_PATH Build the configured independent path plus repeated closure point.
    Q=cfg.path.nodes;
    ns=cfg.path.subintervalsPerSegment;
    if size(Q,2)~=5, error('Expected five closed-path nodes.'); end
    if norm(Q(:,1)-Q(:,5))>1e-14, error('Path must be closed.'); end

    qIndependent=zeros(3,4*ns);
    segment=zeros(1,4*ns);
    localIndex=zeros(1,4*ns);
    c=0;
    for s=1:4
        qa=Q(:,s); qb=Q(:,s+1);
        for j=0:ns-1
            c=c+1;
            t=j/ns;
            qIndependent(:,c)=(1-t)*qa+t*qb;
            segment(c)=s;
            localIndex(c)=j;
        end
    end
    qPlot=[qIndependent qIndependent(:,1)];

    path=struct();
    path.qIndependent=qIndependent;
    path.qPlot=qPlot;
    path.numIndependent=size(qIndependent,2);
    path.numPlot=size(qPlot,2);
    path.segment=segment;
    path.localIndex=localIndex;
    path.segmentBoundaries=1:ns:(4*ns+1);
    path.plotIndex=1:path.numPlot;
end
