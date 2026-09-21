function results = load_exp4_results(mediumName)
%LOAD_EXP4_RESULTS Load a final Experiment-4 result from output/.
    root = setup_exp4();
    mediumName = upper(char(mediumName));
    if ~ismember(mediumName,{'A','B','PAPER'})
        error('mediumName must be A, B, or PAPER.');
    end

    file = fullfile(root,'output',sprintf('Exp4_%s_results.mat',mediumName));
    if exist(file,'file')~=2
        error('Missing %s. Copy the terminal result into output/ or run RUN_%s.',file,mediumName);
    end

    S = load(file,'results');
    if ~isfield(S,'results')
        error('File does not contain variable ''results'': %s',file);
    end
    results = S.results;

    expected = make_config_exp4(mediumName);
    validate_material(results,expected,mediumName,file);
    results.localSourceFile = file;
end

function validate_material(R,cfg,mediumName,file)
    if ~isfield(R,'material') || ~isfield(R.material,'modelType')
        error('Missing material metadata in %s.',file);
    end
    if ~strcmp(char(R.material.modelType),char(cfg.material.modelType))
        error('Material model mismatch in %s.',file);
    end

    if strcmp(mediumName,'A')
        ok = abs(R.material.alpha-cfg.material.alpha)<1e-14 && ...
             abs(R.material.beta-cfg.material.beta)<1e-14 && ...
             abs(R.material.referenceEpsC-cfg.material.referenceEpsC)<1e-14;
    else
        names = {'cB','b0','a12','a34','a56','a135'};
        if strcmp(mediumName,'PAPER'), names{end+1}='axialAmplitude'; end
        ok = true;
        for k=1:numel(names)
            n=names{k};
            ok = ok && isfield(R.material,n) && abs(R.material.(n)-cfg.material.(n))<1e-14;
        end
    end
    if ~ok
        error('The result file %s does not match the final Experiment-4 %s medium.',file,mediumName);
    end
end
