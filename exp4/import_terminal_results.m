function n = import_terminal_results(sourceRoot)
%IMPORT_TERMINAL_RESULTS Copy final Exp4 terminal outputs into local output/.
%   sourceRoot may be the terminal package root or its output directory.
    if nargin<1 || isempty(sourceRoot)
        error('Provide the downloaded terminal package root or output directory.');
    end
    if exist(sourceRoot,'dir')~=7
        error('Source directory does not exist: %s',sourceRoot);
    end

    root = setup_exp4();
    outDir = fullfile(root,'output');
    patterns = {'Exp4_A_*','Exp4_B_*'};
    n = 0;
    for ip=1:numel(patterns)
        hits = dir(fullfile(sourceRoot,'**',patterns{ip}));
        for k=1:numel(hits)
            if hits(k).isdir, continue; end
            src = fullfile(hits(k).folder,hits(k).name);
            dst = fullfile(outDir,hits(k).name);
            copyfile(src,dst);
            n = n+1;
        end
    end
    fprintf('Imported %d Experiment-4 files into %s\n',n,outDir);
end
