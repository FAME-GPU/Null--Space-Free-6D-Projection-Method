function [selected,source] = get_selected_eigenpairs_exp4(cfg,root)
%GET_SELECTED_EIGENPAIRS_EXP4 Load a valid checkpoint or compute it once.

    ckDir=fullfile(root,'checkpoint');
    if ~exist(ckDir,'dir'), mkdir(ckDir); end
    ck=fullfile(ckDir,cfg.output.checkpointFile);
    if cfg.run.reuseSelectedEigenpairCheckpoint && exist(ck,'file')
        S=load(ck,'selected','checkpointSignature');
        if isfield(S,'selected') && isfield(S,'checkpointSignature') && ...
                checkpoint_signature_matches(S.checkpointSignature,cfg)
            selected=S.selected;
            source='checkpoint';
            if cfg.output.verbose
                fprintf('[Exp4] Reusing selected-eigenpair checkpoint: %s\n',ck);
            end
            return
        end
        if cfg.output.verbose
            fprintf('[Exp4] Existing checkpoint signature mismatch; recomputing.\n');
        end
    end

    gpuInfo=gpu_initialize(cfg.gpu);
    fprintf('[Exp4] Building N=5 system and solving tracked modes [%d,%d]...\n', ...
        cfg.tracking.n5Modes(1),cfg.tracking.n5Modes(2));
    sys=build_tracking_system(cfg,cfg.problem.N,gpuInfo);
    selected=solve_tracked_modes_exp4(sys,cfg);
    source='computed';

    if cfg.run.saveSelectedEigenpairCheckpoint
        checkpointSignature=make_checkpoint_signature(cfg); %#ok<NASGU>
        save(ck,'selected','checkpointSignature','-v7.3');
        if cfg.output.verbose
            fprintf('[Exp4] Saved selected-eigenpair checkpoint: %s\n',ck);
        end
    end
    clear sys
    try
        wait(gpuDevice);
    catch
    end
end

function sig=make_checkpoint_signature(cfg)
    sig=struct('N',cfg.problem.N,'q',cfg.problem.q(:).', ...
        'modes',cfg.tracking.n5Modes(:).','epsC',cfg.material.epsC, ...
        'alpha',cfg.material.alpha,'tolCG',cfg.solver.tolCG, ...
        'tolLanczos',cfg.solver.tolLanczos,'tolRecoverM',cfg.solver.tolRecoverM, ...
        'kScan',cfg.scan.N5.k);
end

function tf=checkpoint_signature_matches(sig,cfg)
    ref=make_checkpoint_signature(cfg);
    tf=isstruct(sig) && isequal(sig.N,ref.N) && isequal(sig.modes,ref.modes) && ...
        norm(sig.q-ref.q,inf)<1e-14 && sig.epsC==ref.epsC && sig.alpha==ref.alpha && ...
        sig.tolCG==ref.tolCG && sig.tolLanczos==ref.tolLanczos && ...
        sig.tolRecoverM==ref.tolRecoverM && sig.kScan==ref.kScan;
end
