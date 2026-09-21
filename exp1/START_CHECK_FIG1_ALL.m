    %% Check all Experiment 1 Fig. 1 terminal results
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
exp1bc_startup; reportBC=check_fig1bc_results(); %#ok<NASGU>
