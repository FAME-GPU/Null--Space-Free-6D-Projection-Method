function results = load_experiment3_results_local()
%LOAD_EXPERIMENT3_RESULTS_LOCAL Load Experiment 3 raw results from output/.
% Copy the terminal output files directly into output/, or generate them
% locally with RUN_EXP3.
    root=setup_experiment3_local();
    path=fullfile(root,'output','Experiment3_N5_AllGPU_results.mat');
    if ~exist(path,'file')
        error(['No Experiment3_N5_AllGPU_results.mat was found in output/. ', ...
               'Copy the terminal output files into output/, call ', ...
               'import_terminal_results(sourceDir), or run RUN_EXP3.']);
    end
    S=load(path,'results');
    if ~isfield(S,'results'), error('File does not contain variable results: %s',path); end
    results=S.results;
    fprintf('Loaded Experiment 3 results: %s\n',path);
end
