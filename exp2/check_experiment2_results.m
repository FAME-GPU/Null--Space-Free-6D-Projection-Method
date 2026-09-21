function check_experiment2_results(results,cfg)
%CHECK_EXPERIMENT2_RESULTS Validate downloaded production results before plotting.
    if results.N5.N~=5 || results.N5.NF~=1000000, error('Unexpected N/NF.'); end
    if results.adopted.order~=10 || results.adopted.numCenters~=64, error('Unexpected adopted Taylor setting.'); end
    T=results.tradeoff;
    if height(T)~=18, error('Expected 18 Taylor trade-off rows.'); end
    if ~all(ismember([6 8 10],unique(T.p).')), error('Missing Taylor order.'); end
    if ~all(ismember([1 8 27 64 125 216],unique(T.NumCenters).')), error('Missing center count.'); end
    if any(~isfinite(T.GPUTimeSeconds)) || any(T.GPUTimeSeconds<=0), error('Invalid GPU reconstruction timing.'); end
    if any(~isfinite(T.RelativeFieldErrorMode10)) || any(T.RelativeFieldErrorMode10<0), error('Invalid field error.'); end
    W=results.workflow;
    if height(W)~=2 || any(W.GPUWorkflowTotalSeconds<=0), error('Invalid workflow table.'); end
    if ~isfield(results,'gpu') || ~isfield(results.gpu,'name'), error('Missing GPU metadata.'); end
    fprintf('[Exp2 result check] PASS\n');
    fprintf('  GPU: %s\n',results.gpu.name);
    fprintf('  adopted p=%d, centers=%d, rho=%.6f\n',results.adopted.order,results.adopted.numCenters,results.adopted.rhoMax);
    fprintf('  adopted e2=%.3e, eA=%.3e\n',results.adopted.relativeFieldError,results.adopted.curlCurlActionError);
    fprintf('  GPU totals direct/Taylor = %.3f / %.3f s; ratio %.3fx\n',W.GPUWorkflowTotalSeconds(1),W.GPUWorkflowTotalSeconds(2),results.workflowRatio);
end
