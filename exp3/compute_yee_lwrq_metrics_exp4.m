function [m,spatial] = compute_yee_lwrq_metrics_exp4(fieldPlus,lambda6D,cfg,h,n,saveSpatial)
%COMPUTE_YEE_LWRQ_METRICS_EXP4 All large physical-space kernels on GPU.
%
% For saveSpatial=true, only the z=0 and x=0 relative-deviation slices
% are gathered.  The full 3D rho/deviation field never leaves the GPU.

    if nargin<6, saveSpatial=false; end
    expected=2*n+3;
    if any(size(fieldPlus.u1)~=[expected expected expected])
        error('fieldPlus size does not match requested n=%d.',n);
    end
    if cfg.run.requireGPUPhysicalKernels && ~isa(fieldPlus.u1,'gpuArray')
        error('All-GPU Exp3 received a CPU Yee field.');
    end

    Afield=apply_cropped_A_exp3(fieldPlus,h);
    u1=fieldPlus.u1(2:end-1,2:end-1,2:end-1);
    u2=fieldPlus.u2(2:end-1,2:end-1,2:end-1);
    u3=fieldPlus.u3(2:end-1,2:end-1,2:end-1);

    epsField=sample_scalar_yee_material_exp4(cfg,h,n,u1);
    Bu1=epsField.u1.*u1;
    Bu2=epsField.u2.*u2;
    Bu3=epsField.u3.*u3;
    clear epsField

    a=real(conj(u1).*Afield.u1 + conj(u2).*Afield.u2 + conj(u3).*Afield.u3);
    b=real(conj(u1).*Bu1 + conj(u2).*Bu2 + conj(u3).*Bu3);
    sumB=sum(b(:));
    if gather_scalar(sumB)<=0, error('Cropped Yee mass is not positive.'); end

    lambdaCrop=sum(a(:))/sumB;
    L=compute_lwrq_exp4(a,b,h,cfg,saveSpatial);
    clear a b
    lambda3D=real(L.lambda);
    deltaIdentity=abs(lambda3D-lambdaCrop)/max(abs(lambdaCrop),cast(eps,'like',real(lambdaCrop)));

    nA2=sum(abs(Afield.u1(:)).^2)+sum(abs(Afield.u2(:)).^2)+sum(abs(Afield.u3(:)).^2);
    nB2=sum(abs(Bu1(:)).^2)+sum(abs(Bu2(:)).^2)+sum(abs(Bu3(:)).^2);
    nA=sqrt(nA2); nB=sqrt(nB2);

    nr3sq=residual_sq(Afield.u1,Bu1,lambda3D) + ...
          residual_sq(Afield.u2,Bu2,lambda3D) + ...
          residual_sq(Afield.u3,Bu3,lambda3D);
    eta3D=sqrt(nr3sq)/(nA+abs(lambda3D)*nB);

    lam6=cast(lambda6D,'like',lambda3D);
    nr6sq=residual_sq(Afield.u1,Bu1,lam6) + ...
          residual_sq(Afield.u2,Bu2,lam6) + ...
          residual_sq(Afield.u3,Bu3,lam6);
    eta6D=sqrt(nr6sq)/(nA+abs(lam6)*nB);
    deltaLambda=abs(lambda3D-lam6)/abs(lam6);

    % Gather only compact scalar diagnostics.
    m=struct();
    m.h=h; m.n=n; m.halfWidth=n*h;
    m.lambda6D=lambda6D;
    m.lambdaCrop=gather_scalar(lambdaCrop);
    m.lambda3D=gather_scalar(lambda3D);
    m.eta3D=gather_scalar(eta3D);
    m.etaUsing6DLambda=gather_scalar(eta6D);
    m.deltaLambda=gather_scalar(deltaLambda);
    m.VLWRQ=gather_scalar(L.V);
    m.nuLWRQ=gather_scalar(L.nu);
    m.deltaPart=gather_scalar(L.partitionError);
    m.deltaLWRQIdentity=gather_scalar(deltaIdentity);
    m.croppedMass=gather_scalar(sumB);

    spatial=[];
    if saveSpatial
        if isempty(L.rho)
            error('Internal error: spatial output requested without retaining rho.');
        end
        q=2*n+1;
        c=n+1;

        % Array dimensions follow (x,y,z).  Keep only the two Cartesian
        % planes used by the manuscript: z=0 and x=0.
        z0Dev=abs(L.rho(:,:,c)-lambda3D)./abs(lambda3D);       % rows x, columns y
        x0Rho=squeeze(L.rho(c,:,:));                           % rows y, columns z
        x0Dev=abs(x0Rho-lambda3D)./abs(lambda3D);

        spatial=struct();
        spatial.relativeDeviation_z0=gather(z0Dev);
        spatial.relativeDeviation_x0=gather(x0Dev);
        spatial.coord=(-n:n)*h;
        spatial.lambda3D=m.lambda3D;
        spatial.lambda6D=lambda6D;
        spatial.h=h;
        spatial.n=n;
        spatial.halfWidth=n*h;
        clear z0Dev x0Rho x0Dev
    end
end

function s=residual_sq(Au,Bu,lambda)
    r=Au-lambda*Bu;
    s=sum(abs(r(:)).^2);
    clear r
end
