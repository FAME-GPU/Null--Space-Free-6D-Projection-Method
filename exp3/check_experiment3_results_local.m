function check_experiment3_results_local()
%CHECK_EXPERIMENT3_RESULTS_LOCAL Validate the compact local result structure.
    R=load_experiment3_results_local();
    req={'cfg','eigen','meshCases','meshSummary','windowCases','windowSummary','spatialMaxWindow'};
    for k=1:numel(req)
        if ~isfield(R,req{k}), error('Missing results.%s',req{k}); end
    end

    expectedN=[10 15 20 30 40 60 80 100 120 140 160];
    if ~isequal(R.cfg.window.n,expectedN)
        error('Unexpected Experiment 3 window list.');
    end
    if numel(R.windowSummary)~=11 || numel(R.windowCases)~=22
        error('Expected 11 window summaries and 22 mode/window cases.');
    end

    fprintf('\nExperiment 3 result summary\n');
    fprintf('  modes: %s\n',mat2str(R.eigen.modes));
    fprintf('  lambda6D: %s\n',mat2str(R.eigen.lambda6D,12));
    fprintf('  mesh cases: %d; window cases: %d\n',numel(R.meshCases),numel(R.windowCases));
    fprintf('  maximum window: h=%g, n=%d, L=%g\n', ...
        R.cfg.window.h,R.cfg.window.masterN,R.cfg.window.h*R.cfg.window.masterN);
    if isfield(R,'physicalKernels'), fprintf('  physical kernels: %s\n',R.physicalKernels); end
    if isfield(R,'totalWallTimeSeconds'), fprintf('  total wall time: %.2f min\n',R.totalWallTimeSeconds/60); end

    cfg=R.cfg;
    isDefault=arrayfun(@(s)strcmp(s.study,'mesh') && abs(s.h-cfg.default.h)<1e-14 && s.n==cfg.default.n,R.meshCases);
    D=R.meshCases(isDefault);
    for k=1:numel(D)
        fprintf(['  default mode %d: eta3D=%.3e, nuLWRQ=%.3e, ', ...
                 'deltaLambda=%.3e, identity=%.3e\n'], ...
            D(k).mode,D(k).eta3D,D(k).nuLWRQ,D(k).deltaLambda,D(k).deltaLWRQIdentity);
    end

    for j=1:numel(R.spatialMaxWindow)
        S=R.spatialMaxWindow{j};
        if isempty(S), error('Missing L=4 spatial data for mode %d.',R.eigen.modes(j)); end
        if S.n~=160 || abs(S.halfWidth-4)>1e-12 || ...
                ~isequal(size(S.relativeDeviation_z0),[321 321]) || ...
                ~isequal(size(S.relativeDeviation_x0),[321 321])
            error('Invalid L=4 spatial data for mode %d.',R.eigen.modes(j));
        end
    end
    fprintf('  L=4 z=0/x=0 spatial slices: PASS (321 x 321 for both tracked modes)\n');
end
