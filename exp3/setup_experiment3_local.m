function root = setup_experiment3_local()
%SETUP_EXPERIMENT3_LOCAL Initialize the flat Experiment 3 source directory.
    root=fileparts(mfilename('fullpath'));
    addpath(root);
    names={'checkpoint','output','figure','export'};
    for j=1:numel(names)
        d=fullfile(root,names{j});
        if ~isfolder(d), mkdir(d); end
    end
end
