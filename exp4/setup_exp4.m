function root = setup_exp4()
%SETUP_EXP4 Initialize the flat Experiment 4 source directory.
    root=fileparts(mfilename('fullpath'));
    addpath(root);
    names={'output','checkpoint','figure'};
    for j=1:numel(names)
        d=fullfile(root,names{j});
        if ~isfolder(d), mkdir(d); end
    end
end
