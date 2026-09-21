%EXP1BC_STARTUP Initialize Experiment-1 Fig.1(b,c) project.
root = exp1bc_project_root();
addpath(root);
cd(root);
cfgStartup = make_config_exp1bc();
outDirs = {fullfile(root,cfgStartup.output.resultsDirectory), ...
           fullfile(root,cfgStartup.output.checkpointDirectory), ...
           fullfile(root,cfgStartup.output.figureDirectory), ...
           fullfile(root,'terminal_results')};
for kStartup = 1:numel(outDirs)
    if ~exist(outDirs{kStartup},'dir'), mkdir(outDirs{kStartup}); end
end
clear root cfgStartup outDirs kStartup
