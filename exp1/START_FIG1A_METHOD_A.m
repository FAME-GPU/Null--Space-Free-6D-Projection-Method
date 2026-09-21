%% Experiment 1 Fig. 1(a) -- Method A
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
results=run_fig1a_A_only(); %#ok<NASGU>
