function dest = import_terminal_results(sourceDir)
%IMPORT_TERMINAL_RESULTS Import an Experiment 3 terminal output set locally.
    root=setup_experiment3_local();
    if nargin<1 || isempty(sourceDir)
        sourceDir=uigetdir(pwd,'Select Experiment 3 terminal result directory');
        if isequal(sourceDir,0), error('No source directory selected.'); end
    end
    if ~isfolder(sourceDir), error('Source directory does not exist: %s',sourceDir); end

    hits=dir(fullfile(sourceDir,'**','Experiment3_N5_AllGPU_results.mat'));
    if isempty(hits)
        error('Experiment3_N5_AllGPU_results.mat was not found below %s.',sourceDir);
    end
    [~,ord]=sort([hits.datenum],'descend'); hits=hits(ord);
    srcOut=hits(1).folder;
    if numel(hits)>1
        fprintf('Multiple result MAT files found; importing newest: %s\n',fullfile(srcOut,hits(1).name));
    end

    dest=fullfile(root,'output');
    if ~exist(dest,'dir'), mkdir(dest); end
    patterns={'Experiment3_N5_AllGPU_results.mat','Experiment3_N5_*.csv','Experiment3_N5_*.txt'};
    for ip=1:numel(patterns)
        F=dir(fullfile(srcOut,patterns{ip}));
        for k=1:numel(F)
            copyfile(fullfile(F(k).folder,F(k).name),fullfile(dest,F(k).name));
            fprintf('Imported: %s\n',F(k).name);
        end
    end

    % Import the validated legacy-named selected-eigenpair checkpoint if present.
    chk=dir(fullfile(sourceDir,'**','Experiment4_N5_selected_eigenpairs_checkpoint.mat'));
    if ~isempty(chk)
        [~,ii]=max([chk.datenum]);
        chkDest=fullfile(root,'checkpoint');
        copyfile(fullfile(chk(ii).folder,chk(ii).name),fullfile(chkDest,chk(ii).name));
        fprintf('Imported optional checkpoint: %s\n',chk(ii).name);
    end
end
