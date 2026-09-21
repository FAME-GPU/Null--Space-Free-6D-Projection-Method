function [m,spatial] = compute_yee_lwrq_metrics_exp4(fieldPlus,lambda6D,cfg,st,h,n,saveSpatial)
%COMPUTE_YEE_LWRQ_METRICS_EXP4 GPU 3D residual/LWRQ/spectral closure.
    if nargin<7, saveSpatial=false; end
    expected=2*n+3;
    if any(size(fieldPlus.u1)~=[expected expected expected])
        error('fieldPlus size does not match requested n=%d.',n);
    end
    if ~isa(fieldPlus.u1,'gpuArray')
        error('Yee/LWRQ production postprocessing must remain on GPU.');
    end

    Afield=apply_cropped_A_exp3(fieldPlus,h);
    u1=fieldPlus.u1(2:end-1,2:end-1,2:end-1);
    u2=fieldPlus.u2(2:end-1,2:end-1,2:end-1);
    u3=fieldPlus.u3(2:end-1,2:end-1,2:end-1);

    epsField=sample_scalar_yee_material_exp4(cfg,st,h,n,u1);
    Bu1=epsField.u1.*u1; Bu2=epsField.u2.*u2; Bu3=epsField.u3.*u3;

    a=real(conj(u1).*Afield.u1 + conj(u2).*Afield.u2 + conj(u3).*Afield.u3);
    b=real(conj(u1).*Bu1 + conj(u2).*Bu2 + conj(u3).*Bu3);
    croppedMass=sum(b(:));
    if gather_scalar(croppedMass)<=0, error('Cropped Yee mass is not positive.'); end

    lambdaCrop=sum(a(:))./croppedMass;
    L=compute_lwrq_exp4(a,b,h,cfg);
    lambda3D=real(L.lambda);
    tiny=cast(eps('double'),'like',lambda3D);
    deltaIdentity=abs(lambda3D-lambdaCrop)./max(abs(lambdaCrop),tiny);

    nA=sqrt(sum(abs(Afield.u1(:)).^2)+sum(abs(Afield.u2(:)).^2)+sum(abs(Afield.u3(:)).^2));
    nB=sqrt(sum(abs(Bu1(:)).^2)+sum(abs(Bu2(:)).^2)+sum(abs(Bu3(:)).^2));
    r1=Afield.u1-lambda3D.*Bu1;
    r2=Afield.u2-lambda3D.*Bu2;
    r3=Afield.u3-lambda3D.*Bu3;
    nr=sqrt(sum(abs(r1(:)).^2)+sum(abs(r2(:)).^2)+sum(abs(r3(:)).^2));
    eta3D=nr./(nA+abs(lambda3D).*nB);

    lambda6DGPU=cast(lambda6D,'like',lambda3D);
    m=struct();
    m.h=h; m.n=n; m.halfWidth=n*h;
    m.lambda6D=lambda6DGPU;
    m.lambdaCrop=lambdaCrop;
    m.lambda3D=lambda3D;
    m.lambdaLWRQ=lambda3D;
    m.eta3D=eta3D;
    m.deltaLambda=abs(lambda3D-lambda6DGPU)./abs(lambda6DGPU);
    m.VLWRQ=L.V;
    m.nuLWRQ=L.nu;
    m.partitionError=L.partitionError;
    m.lwrqIdentityError=deltaIdentity;
    m.croppedMass=croppedMass;

    spatial=[];
    if saveSpatial
        coordGPU=cast((-n:n).*h,'like',u1);
        spatial=struct();
        spatial.rho=L.rho;
        spatial.relativeDeviation=abs(L.rho-lambda3D)./abs(lambda3D);
        spatial.intensity=abs(u1).^2+abs(u2).^2+abs(u3).^2;
        spatial.coord=coordGPU;
    end
end
