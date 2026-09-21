function budget = make_run_budget(iterLimit,timeLimit)
%MAKE_RUN_BUDGET Shared iteration/time budget with controlled-stop errors.
%
% The timer starts when this function is called. Iterations are accumulated
% explicitly by the instrumented linear solvers. Exactly hitting ITERLIMIT is
% allowed if the current solve has converged; any further linear iteration is
% forbidden. Time is checked at safe solver boundaries.

    if nargin<1 || isempty(iterLimit), iterLimit=Inf; end
    if nargin<2 || isempty(timeLimit), timeLimit=Inf; end
    iterUsed = 0;
    t0 = tic;

    budget.addIterations = @addIterations;
    budget.check = @check;
    budget.checkCanContinue = @checkCanContinue;
    budget.capMaxit = @capMaxit;
    budget.remainingIterations = @remainingIterations;
    budget.elapsedSeconds = @elapsedSeconds;
    budget.snapshot = @snapshot;

    function addIterations(n,label)
        n = double(n);
        if ~(isscalar(n) && isfinite(n) && n>=0)
            error('EXP1:BudgetInternal','Invalid iteration increment.');
        end
        iterUsed = iterUsed + n;
        if iterUsed > iterLimit
            error('EXP1:IterationLimit', ...
                'Iteration budget exceeded after %s: %.0f > %.0f.',label,iterUsed,iterLimit);
        end
        checkTime(label);
    end

    function check(label)
        checkTime(label);
        if iterUsed > iterLimit
            error('EXP1:IterationLimit','Iteration budget exceeded at %s.',label);
        end
    end

    function checkCanContinue(label)
        checkTime(label);
        if isfinite(iterLimit) && iterUsed >= iterLimit
            error('EXP1:IterationLimit', ...
                'Iteration budget exhausted at %s: %.0f / %.0f.',label,iterUsed,iterLimit);
        end
    end

    function m = capMaxit(requested)
        checkTime('linear-solver entry');
        rem = remainingIterations();
        if rem <= 0
            error('EXP1:IterationLimit','No linear-iteration budget remains.');
        end
        if isinf(rem)
            m = requested;
        else
            m = min(double(requested),floor(rem));
        end
        m = max(1,round(m));
    end

    function r = remainingIterations()
        r = iterLimit - iterUsed;
    end

    function t = elapsedSeconds()
        t = toc(t0);
    end

    function s = snapshot()
        s = struct('iterationLimit',iterLimit,'timeLimitSeconds',timeLimit, ...
            'iterationsUsed',iterUsed,'elapsedSeconds',toc(t0), ...
            'remainingIterations',iterLimit-iterUsed);
    end

    function checkTime(label)
        t = toc(t0);
        if t > timeLimit
            error('EXP1:TimeLimit', ...
                'Solver wall-time budget exceeded at %s: %.3f s > %.3f s.', ...
                label,t,timeLimit);
        end
    end
end
