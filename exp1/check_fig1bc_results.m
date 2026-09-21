function check_fig1bc_results()
%CHECK_FIG1BC_RESULTS Cross-method consistency checks after all runs.
    cfg=make_config_exp1bc();
    root=exp1bc_project_root();
    D=resolve_fig1bc_result_dir();
    methods={'A','B','C'}; S=cell(3,1);
    for j=1:3
        f=fullfile(D,sprintf('fig1bc_%s.mat',methods{j}));
        if ~exist(f,'file'), error('Missing %s.',f); end
        T=load(f,'results'); S{j}=T.results;
    end
    fprintf('Cross-method Fig. 1(b,c) check\n');
    for ii=1:numel(cfg.scale.NList)
        N=cfg.scale.NList(ii); ok=true(3,1);
        for j=1:3, ok(j)=strcmp(S{j}.scaleResults{ii}.status,'success'); end
        if nnz(ok)>=2
            ids=find(ok); base=S{ids(1)}.scaleResults{ii}.lambda(:);
            for jj=2:numel(ids)
                cur=S{ids(jj)}.scaleResults{ii}.lambda(:);
                rel=max(abs(cur-base)./max(abs(base),realmin));
                fprintf('  N=%d: Method %s vs %s max relative eigenvalue difference %.3e\n', ...
                    N,methods{ids(1)},methods{ids(jj)},rel);
                if rel>1e-7
                    warning('N=%d cross-method eigenvalue discrepancy %.3e exceeds 1e-7.',N,rel);
                end
            end
        end
    end
    fprintf('Result check finished.\n');
end
