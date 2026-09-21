    %% Experiment 1 merged package self-check
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

self_check_fig1a_package();
self_check_exp1bc();
fprintf('Merged Experiment-1 self-check PASSED.\n');

