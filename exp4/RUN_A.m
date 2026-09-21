function results = RUN_A()
%RUN_A Local/GPU production entry point for medium A.
    setup_exp4();
    results = run_exp4_medium('A');
end
