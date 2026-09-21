function results = RUN_EXP3()
%RUN_EXP3 Primary Experiment 3 entry point.
%   Terminal: matlab -batch "RUN_EXP3"
%   Runs the exact production data path used for Fig. 3 and writes all
%   numerical results to <root>/output/. Plotting is deliberately separate.

    root = setup_experiment3_local();
    logFile = fullfile(root,'Exp3_run.log');
    if exist(logFile,'file'), delete(logFile); end
    diary(logFile);
    cleanupDiary = onCleanup(@()safe_diary_off()); %#ok<NASGU>

    fprintf('Experiment 3 FINAL production run\n');
    fprintf('Root: %s\n',root);
    fprintf('Start: %s\n\n',datestr(now,31));

    try
        cfg = make_config_experiment4();
        results = run_experiment4(cfg);
        fprintf('\nFinish: %s\n',datestr(now,31));
    catch ME
        fprintf(2,'\nExperiment 3 failed: %s\n',ME.message);
        fprintf(2,'%s\n',getReport(ME,'extended','hyperlinks','off'));
        rethrow(ME);
    end
end

function safe_diary_off()
    try
        diary off;
    catch
    end
end
