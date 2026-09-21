function files = extract_experiment3_paper_data(R)
%EXTRACT_EXPERIMENT3_PAPER_DATA Export compact manuscript-facing tables.
    if nargin<1 || isempty(R), R=load_experiment3_results_local(); end
    root=setup_experiment3_local(); out=fullfile(root,'export');
    if ~exist(out,'dir'), mkdir(out); end

    cfg=R.cfg;
    idx=arrayfun(@(s)strcmp(s.study,'mesh') && abs(s.h-cfg.default.h)<1e-14 && s.n==cfg.default.n,R.meshCases);
    D=R.meshCases(idx);
    Tdefault=table([D.mode].',[D.lambda6D].',[D.lambda3D].',[D.eta3D].',[D.nuLWRQ].',[D.deltaLambda].', ...
        [D.deltaPart].',[D.deltaLWRQIdentity].', ...
        'VariableNames',{'mode','lambda6D','lambda3D','eta3D','nuLWRQ','deltaLambda','partitionError','lwrqIdentityError'});

    M=R.meshSummary;
    Tmesh=table([M.h].',[M.n].',[M.halfWidth].',[M.maxEta3D].',[M.maxNuLWRQ].',[M.maxDeltaLambda].', ...
        [M.numCenters].',[M.phaseRadiusBound].',[M.reconstructionTimeSeconds].', ...
        'VariableNames',{'h','n','halfWidth','maxEta3D','maxNuLWRQ','maxDeltaLambda','numCenters','phaseRadiusBound','reconstructionTimeSeconds'});

    W=R.windowSummary;
    Twindow=table([W.h].',[W.n].',[W.halfWidth].',[W.maxEta3D].',[W.maxNuLWRQ].',[W.maxDeltaLambda].', ...
        'VariableNames',{'h','n','halfWidth','maxEta3D','maxNuLWRQ','maxDeltaLambda'});

    files={fullfile(out,'Experiment3_Paper_DefaultModes.csv'), ...
           fullfile(out,'Experiment3_Paper_MeshSensitivity.csv'), ...
           fullfile(out,'Experiment3_Paper_WindowSensitivity.csv'), ...
           fullfile(out,'Experiment3_Paper_Data.mat')};
    writetable(Tdefault,files{1}); writetable(Tmesh,files{2}); writetable(Twindow,files{3});
    save(files{4},'Tdefault','Tmesh','Twindow','-v7.3');
    fprintf('Exported Experiment 3 manuscript data to %s\n',out);
end
