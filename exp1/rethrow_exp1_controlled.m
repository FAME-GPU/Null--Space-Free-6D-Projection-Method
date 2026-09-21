function rethrow_exp1_controlled(ME,context,budget)
%RETHROW_EXP1_CONTROLLED Preserve controlled status through solver wrappers.
% If MATLAB drops the underlying cause when wrapping a user-function error,
% the shared budget snapshot still identifies exhausted iteration/time caps.
    if nargin<2 || isempty(context), context='iterative solver'; end
    if nargin>=3 && ~isempty(budget)
        s=budget.snapshot();
        if isfinite(s.timeLimitSeconds) && s.elapsedSeconds>s.timeLimitSeconds
            error('EXP1:TimeLimit','%s stopped by wall-time budget. Wrapped message: %s',context,ME.message);
        end
        if isfinite(s.iterationLimit) && s.iterationsUsed>=s.iterationLimit
            error('EXP1:IterationLimit','%s stopped by iteration budget. Wrapped message: %s',context,ME.message);
        end
    end
    [status,controlled]=classify_exp1_exception(ME);
    if ~controlled, rethrow(ME); end
    switch status
        case 'iteration_limit'
            error('EXP1:IterationLimit','%s stopped by iteration budget. Wrapped message: %s',context,ME.message);
        case 'time_limit'
            error('EXP1:TimeLimit','%s stopped by wall-time budget. Wrapped message: %s',context,ME.message);
        case 'memory_limit'
            error('EXP1:MemoryLimit','%s stopped by memory exhaustion. Wrapped message: %s',context,ME.message);
        case 'not_converged'
            error('EXP1:NotConverged','%s did not converge. Wrapped message: %s',context,ME.message);
        otherwise
            rethrow(ME)
    end
end
