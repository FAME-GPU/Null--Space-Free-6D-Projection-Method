function assert_gpu_basis_alive_exp3(basis,context)
%ASSERT_GPU_BASIS_ALIVE_EXP3 Verify that persistent Taylor GPU arrays still exist.
%
% A gpuArray variable can remain in the MATLAB workspace after its device
% storage has been invalidated by a GPU reset.  isa(...,'gpuArray') alone is
% therefore not sufficient.  This helper checks both type and device
% validity and produces an explicit diagnostic before an expensive step.

    if nargin<2 || isempty(context)
        context='Experiment 3';
    end

    if ~isstruct(basis) || ~isfield(basis,'qModes') || ~isfield(basis,'Qalpha')
        error('%s: Taylor basis is incomplete.',context);
    end
    if ~isa(basis.qModes,'gpuArray') || ~isa(basis.Qalpha,'gpuArray')
        error('%s: Taylor basis is not resident on the GPU.',context);
    end

    % existsOnGPU is the correct check after a possible device reset.  A
    % fallback touch is kept for compatibility with MATLAB releases where
    % the helper is unavailable.
    qAlive=false;
    aAlive=false;
    usedExistsOnGPU=false;
    if exist('existsOnGPU','file')~=0 || exist('existsOnGPU','builtin')~=0
        try
            qAlive=logical(existsOnGPU(basis.qModes));
            aAlive=logical(existsOnGPU(basis.Qalpha));
            usedExistsOnGPU=true;
        catch
            % Some MATLAB releases expose existsOnGPU only for selected
            % GPU object classes.  Fall through to a one-element device
            % touch, which also detects invalidated gpuArray storage.
        end
    end
    if ~usedExistsOnGPU
        try
            gather(basis.qModes(1));
            gather(basis.Qalpha(1));
            qAlive=true;
            aAlive=true;
        catch
            qAlive=false;
            aAlive=false;
        end
    end

    if ~(qAlive && aAlive)
        error(['%s: reusable Taylor GPU basis has been invalidated. ', ...
               'The active GPU was most likely reset after the basis was built. ', ...
               'Do not call gpuDevice(index) after persistent gpuArray data exist.'],context);
    end
end
