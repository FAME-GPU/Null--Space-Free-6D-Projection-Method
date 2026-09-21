function destination = plot_experiment(id,source)
%PLOT_EXPERIMENT Plot locally computed results; this repository ships no data.
%   plot_experiment(1) plots Fig. 1(a-c) after all six Exp. 1 jobs complete.
%   plot_experiment(2) plots Fig. 2(a-b) after both Exp. 2 jobs complete.
%   plot_experiment(3) plots Fig. 3(a-b).
%   plot_experiment(4) plots Fig. 4(a-c) for the PAPER medium.
%   The optional second argument 'computed' is accepted for compatibility.
    validateattributes(id,{'numeric'},{'scalar','integer','>=',1,'<=',4});
    if nargin>1
        assert(strcmp(char(source),'computed'), ...
            'No reference data are bundled. Run run_experiment(id) first.');
    end
    repo=fileparts(mfilename('fullpath'));
    root=fullfile(repo,sprintf('exp%d',id));
    ctx=experiment_context(repo,root); %#ok<NASGU>
    switch id
        case 1
            files={'fig1a_A.mat','fig1a_B.mat','fig1a_C.mat', ...
                   'fig1bc_A.mat','fig1bc_B.mat','fig1bc_C.mat'};
            dataDir=fullfile(root,'results'); figureDir='figures';
        case 2
            files={'Experiment2_N5_n40_AllGPU_results.mat', ...
                   'Experiment2b_N5_NgScaling_results.mat'};
            dataDir=fullfile(root,'output'); figureDir='figures';
        case 3
            files={'Experiment3_N5_AllGPU_results.mat'};
            dataDir=fullfile(root,'output'); figureDir='figure';
        case 4
            files={'Exp4_PAPER_results.mat'};
            dataDir=fullfile(root,'output'); figureDir='figure';
    end
    for j=1:numel(files)
        f=fullfile(dataDir,files{j});
        if ~isfile(f)
            error('maxwell:MissingResults', ...
                'Missing %s. Compute the required data with run_experiment(%d) first.',f,id);
        end
    end
    switch id
        case 1
            plot_fig1a_final(); plot_fig1bc();
        case 2
            PLOT_FIG2A_FROM_MAT(); PLOT_FIG2B_FROM_MAT();
        case 3
            PLOT_EXP3();
        case 4
            PLOT_PAPER();
    end
    destination=fullfile(root,figureDir);
    fprintf('Figures exported to %s\n',destination);
end
