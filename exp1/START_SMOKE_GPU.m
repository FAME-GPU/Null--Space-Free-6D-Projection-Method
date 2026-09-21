root = fileparts(mfilename('fullpath')); addpath(root); cd(root); clear root;
exp1bc_startup; smoke_test_rank1_lowmem();
