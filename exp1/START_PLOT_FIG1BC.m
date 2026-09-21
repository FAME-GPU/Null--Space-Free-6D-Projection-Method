%% Plot Experiment 1 Fig. 1(b,c)
clearvars; clc;
st = dbstack('-completenames');
if ~isempty(st) && isfield(st,'file') && ~isempty(st(1).file)
    root = fileparts(st(1).file);
else
    root = pwd;
end
addpath(root);
cd(root);
clear st root;
exp1bc_startup; plot_fig1bc();
