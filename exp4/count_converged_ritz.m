function [nconv,relativeResiduals] = count_converged_ritz(residuals,values,tol)
%COUNT_CONVERGED_RITZ Compare each residual with its own Ritz value.
    residuals=residuals(:); values=values(:);
    assert(numel(residuals)==numel(values),'Ritz array sizes differ.');
    threshold=tol*max(eps('double')^(2/3),abs(values));
    nconv=nnz(isfinite(residuals) & isfinite(values) & residuals<threshold);
    relativeResiduals=residuals./max(abs(values),eps('double')^(2/3));
end
