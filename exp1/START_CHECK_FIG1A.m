%% Check Experiment 1 Fig. 1(a) terminal results
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
reportA=check_fig1a_results(); %#ok<NASGU>
