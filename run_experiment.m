function run_experiment(id,part)
%RUN_EXPERIMENT Compute one experiment on a supported NVIDIA GPU.
%   run_experiment(1,'1a-A')  % 1a-A/B/C or 1bc-A/B/C, default 'all'
%   run_experiment(2,'2a')    % 2a, 2b, or all
%   run_experiment(3)         % selected modes + physical diagnostics
%   run_experiment(4)         % manuscript equation (6.2), 'PAPER'
%   run_experiment(4,'A') or (4,'B') runs a legacy medium with corrected solver.
    validateattributes(id,{'numeric'},{'scalar','integer','>=',1,'<=',4});
    if nargin<2, part='all'; end
    part=char(part);
    repo=fileparts(mfilename('fullpath'));
    root=fullfile(repo,sprintf('exp%d',id));
    if id==1, valid={'all','1a-A','1a-B','1a-C','1bc-A','1bc-B','1bc-C'};
    elseif id==2, valid={'all','2a','2b'};
    elseif id==3, valid={'all'};
    else, valid={'all','PAPER','A','B'};
    end
    assert(any(strcmp(part,valid)),'Unsupported experiment part. See help run_experiment.');
    ctx=experiment_context(repo,root); %#ok<NASGU>
    assert(license('test','Distrib_Computing_Toolbox'), ...
        'Production requires Parallel Computing Toolbox and a supported NVIDIA GPU.');
    gpuDevice();
    fprintf('Experiment %d / %s: computed data go to %s\n',id,part,root);
    switch id
        case 1
            for name={'results','checkpoints','figures'}
                if ~isfolder(name{1}), mkdir(name{1}); end
            end
            for method='ABC'
                if strcmp(part,'all') || strcmp(part,['1a-' method])
                    if method=='A', run_fig1a_A_only();
                    else, run_fig1a_BC_method(method); end
                end
                if strcmp(part,'all') || strcmp(part,['1bc-' method])
                    run_fig1bc_method(method);
                end
            end
        case 2
            if strcmp(part,'all') || strcmp(part,'2a'), RUN_EXPERIMENT2_FIG2A(); end
            if strcmp(part,'all') || strcmp(part,'2b'), RUN_EXPERIMENT2_FIG2B(); end
        case 3
            RUN_EXP3();
        case 4
            if strcmp(part,'all'), part='PAPER'; end
            run_exp4_medium(part);
    end
end
