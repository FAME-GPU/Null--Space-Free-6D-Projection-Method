function cleanup = experiment_context(repo,experimentRoot)
%EXPERIMENT_CONTEXT Activate one flat experiment and restore caller state.
    oldPath=path; oldDir=pwd;
    cleanup=onCleanup(@()restore_context(oldPath,oldDir));
    entries=strsplit(path,pathsep);
    for k=1:4
        root=fullfile(repo,sprintf('exp%d',k));
        for j=1:numel(entries)
            entry=entries{j};
            if strcmp(entry,root) || startsWith(entry,[root filesep])
                rmpath(entry);
            end
        end
    end
    addpath(repo);
    cd(experimentRoot);
    addpath(experimentRoot,'-begin');
    rehash;
end
function restore_context(p,d)
    cd(d); path(p); rehash;
end
