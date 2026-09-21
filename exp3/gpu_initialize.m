function gpuInfo = gpu_initialize(gpuOpt)
%GPU_INITIALIZE Validate and initialize the requested MATLAB GPU device.
%
% gpuOpt fields:
%   useGPU      : must be true for this GPU version
%   deviceIndex : positive integer, default 1
%   resetDevice : reset the selected device before use, default false
%   precision   : 'double' or 'single', default 'double'
%
% This initializer is deliberately idempotent when the requested GPU is
% already active and resetDevice=false.  In that common case it uses
% gpuDevice with no index, avoiding an unnecessary device reset that would
% invalidate existing gpuArray objects.

    if nargin < 1 || isempty(gpuOpt)
        gpuOpt = struct();
    end

    if ~isfield(gpuOpt,'useGPU') || isempty(gpuOpt.useGPU)
        gpuOpt.useGPU = true;
    end
    if ~gpuOpt.useGPU
        error('This package is the GPU version. Set gpuOpt.useGPU = true.');
    end

    if ~isfield(gpuOpt,'deviceIndex') || isempty(gpuOpt.deviceIndex)
        gpuOpt.deviceIndex = 1;
    end
    if ~isfield(gpuOpt,'resetDevice') || isempty(gpuOpt.resetDevice)
        gpuOpt.resetDevice = false;
    end
    if ~isfield(gpuOpt,'precision') || isempty(gpuOpt.precision)
        gpuOpt.precision = 'double';
    end

    precision = lower(char(gpuOpt.precision));
    if ~strcmp(precision,'double') && ~strcmp(precision,'single')
        error('gpuOpt.precision must be ''double'' or ''single''.');
    end

    if exist('gpuDeviceCount','file') == 0
        error(['Parallel Computing Toolbox is required. ', ...
               'MATLAB cannot find gpuDeviceCount.']);
    end

    count = gpuDeviceCount;
    if count < gpuOpt.deviceIndex
        error('Requested GPU %d, but only %d supported GPU(s) are available.', ...
            gpuOpt.deviceIndex,count);
    end

    % Query the current device without resetting it.  Only select by index
    % when a different device is genuinely requested.
    g = gpuDevice;
    if isprop(g,'Index') && g.Index ~= gpuOpt.deviceIndex
        g = gpuDevice(gpuOpt.deviceIndex); %#ok<GPUDEV> intentional device selection at initialization
    end

    if gpuOpt.resetDevice
        reset(g);
        g = gpuDevice; % query only; do not reselect/reset a second time
    end

    gpuInfo = struct();
    gpuInfo.deviceIndex = gpuOpt.deviceIndex;
    gpuInfo.name = g.Name;
    gpuInfo.computeCapability = g.ComputeCapability;
    if isnumeric(gpuInfo.computeCapability)
        gpuInfo.computeCapability = num2str(gpuInfo.computeCapability);
    end
    gpuInfo.totalMemory = g.TotalMemory;
    gpuInfo.availableMemoryAtStart = g.AvailableMemory;
    gpuInfo.precision = precision;
    gpuInfo.resetDevice = logical(gpuOpt.resetDevice);
    gpuInfo.useGPU = true;

    fprintf('GPU device %d: %s\n',gpuInfo.deviceIndex,gpuInfo.name);
    fprintf('  compute capability = %s\n',gpuInfo.computeCapability);
    fprintf('  total memory       = %.3f GiB\n',gpuInfo.totalMemory/2^30);
    fprintf('  available memory   = %.3f GiB\n',gpuInfo.availableMemoryAtStart/2^30);
    fprintf('  working precision  = %s\n',gpuInfo.precision);
end
