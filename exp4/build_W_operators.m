function [Wop,WHop] = build_W_operators(W_1_diag,W_2_diag,n)
%BUILD_W_OPERATORS GPU-compatible diagonal W and W^* handles.

    W_1_diag = W_1_diag(:);
    W_2_diag = W_2_diag(:);
    if numel(W_1_diag)~=n || numel(W_2_diag)~=n
        error('W diagonal lengths must equal n.');
    end

    Wop = @applyW;
    WHop = @applyWH;

    function y = applyW(x)
        if size(x,1)~=n, error('Wop input must have n rows.'); end
        y = [W_1_diag.*x;W_2_diag.*x];
    end

    function y = applyWH(x)
        if size(x,1)~=2*n, error('WHop input must have 2*n rows.'); end
        y = conj(W_1_diag).*x(1:n,:) + conj(W_2_diag).*x(n+1:2*n,:);
    end
end
