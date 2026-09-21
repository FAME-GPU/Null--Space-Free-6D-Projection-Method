function report = check_fig1a_results()
%CHECK_FIG1A_RESULTS Check A/B/C terminal results before plotting Fig. 1(a).
    root=fileparts(which('START_CHECK_FIG1A')); if isempty(root),root=pwd;end
    td=fullfile(root,'results');
    methods='ABC'; R=cell(3,1);
    for j=1:3
        f=fullfile(td,sprintf('fig1a_%c.mat',methods(j)));
        if ~exist(f,'file'), error('Missing %s.',f); end
        S=load(f,'results');
        if ~isfield(S,'results') || ~isfield(S.results,'qResults') || numel(S.results.qResults)~=12
            error('Unexpected result structure in %s.',f);
        end
        R{j}=S.results;
    end

    maxRelAB=0; maxRelAC=0; maxRelBC=0;
    for iq=1:12
        a=R{1}.qResults{iq}.lambda(:);
        b=R{2}.qResults{iq}.lambda(:);
        c=R{3}.qResults{iq}.lambda(:);
        if numel(a)~=10 || numel(b)~=10 || numel(c)~=10
            error('Expected 10 eigenvalues at q%02d.',iq);
        end
        maxRelAB=max(maxRelAB,max(abs(a-b)./max(abs(a),realmin)));
        maxRelAC=max(maxRelAC,max(abs(a-c)./max(abs(a),realmin)));
        maxRelBC=max(maxRelBC,max(abs(b-c)./max(abs(b),realmin)));
    end

    report=struct('maxRelativeAB',maxRelAB,'maxRelativeAC',maxRelAC,'maxRelativeBC',maxRelBC);
    fprintf('Fig. 1(a) A/B/C spectrum check:\n');
    fprintf('  max relative A-B = %.6e\n',maxRelAB);
    fprintf('  max relative A-C = %.6e\n',maxRelAC);
    fprintf('  max relative B-C = %.6e\n',maxRelBC);
end
