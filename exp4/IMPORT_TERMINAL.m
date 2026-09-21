function n = IMPORT_TERMINAL(sourceRoot)
%IMPORT_TERMINAL Convenience entry for copying terminal outputs locally.
    setup_exp4();
    n = import_terminal_results(sourceRoot);
end
