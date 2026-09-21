function y=gather_exp4_for_io(x)
%GATHER_EXP4_FOR_IO Recursively gather GPU arrays only for serialization.
    if isa(x,'gpuArray')
        y=gather(x);
    elseif isstruct(x)
        y=x;
        f=fieldnames(x);
        for k=1:numel(x)
            for j=1:numel(f)
                y(k).(f{j})=gather_exp4_for_io(x(k).(f{j}));
            end
        end
    elseif iscell(x)
        y=cell(size(x));
        for j=1:numel(x), y{j}=gather_exp4_for_io(x{j}); end
    else
        y=x;
    end
end
