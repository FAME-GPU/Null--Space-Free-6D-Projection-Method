function rootDir = setup_experiment2_package()
%SETUP_EXPERIMENT2_PACKAGE Initialize the flat Experiment 2 source directory.
    rootDir=fileparts(mfilename('fullpath'));
    addpath(rootDir);
    names={'output','figures'};
    for j=1:numel(names)
        d=fullfile(rootDir,names{j});
        if ~isfolder(d), mkdir(d); end
    end
end
