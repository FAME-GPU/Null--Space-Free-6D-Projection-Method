function [status,isControlled] = classify_exp1_exception(ME)
%CLASSIFY_EXP1_EXCEPTION Classify direct or wrapped MATLAB solver errors.
% PCG may wrap errors raised by a user operator as "User supplied function
% failed". Search the full MException cause tree so budget/OOM conditions
% keep their intended status.

    [status,isControlled]=classify_one(ME);
    if isControlled, return; end
    try
        causes=ME.cause;
    catch
        causes={};
    end
    for j=1:numel(causes)
        [st,ctrl]=classify_exp1_exception(causes{j});
        if ctrl, status=st; isControlled=true; return; end
    end
end

function [status,isControlled]=classify_one(ME)
    id=char(ME.identifier); lid=lower(id); msg=lower(char(ME.message));
    if strcmp(id,'EXP1:IterationLimit')
        status='iteration_limit'; isControlled=true;
    elseif strcmp(id,'EXP1:TimeLimit')
        status='time_limit'; isControlled=true;
    elseif strcmp(id,'EXP1:MemoryLimit')
        status='memory_limit'; isControlled=true;
    elseif strcmp(id,'EXP1:NotConverged')
        status='not_converged'; isControlled=true;
    elseif contains(msg,'out of memory') || contains(msg,'insufficient memory') || ...
            contains(msg,'cuda_error_out_of_memory') || contains(msg,'内存不足') || ...
            contains(msg,'显存不足') || contains(msg,'无法分配') || ...
            contains(lid,'outofmemory') || contains(lid,'nomem') || contains(lid,'oom') || ...
            (contains(msg,'gpu') && contains(msg,'memory'))
        status='memory_limit'; isControlled=true;
    else
        status='runtime_error'; isControlled=false;
    end
end
