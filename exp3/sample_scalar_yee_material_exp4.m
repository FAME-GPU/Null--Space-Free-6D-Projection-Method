function epsField = sample_scalar_yee_material_exp4(cfg,h,n,prototype)
%SAMPLE_SCALAR_YEE_MATERIAL_EXP4 Scalar epsilon on Yee edges, on device.
    if nargin<4 || isempty(prototype), prototype=0; end
    idx=cast((-n:n)*h,'like',prototype);
    P=cast(cfg.problem.P,'like',prototype);
    epsField=struct();
    epsField.u1=eval_component(idx+h/2,idx,idx,P,cfg.material,prototype);
    epsField.u2=eval_component(idx,idx+h/2,idx,P,cfg.material,prototype);
    epsField.u3=eval_component(idx,idx,idx+h/2,P,cfg.material,prototype);
end

function e=eval_component(x,y,z,P,mat,prototype)
    X=reshape(x,[],1,1); Y=reshape(y,1,[],1); Z=reshape(z,1,1,[]);
    m=numel(x);
    S=zeros(m,m,m,'like',prototype);
    for ell=1:size(P,1)
        phase=P(ell,1).*X+P(ell,2).*Y+P(ell,3).*Z;
        S=S+cos(phase);
    end
    e=cast(mat.epsC,'like',prototype).*exp(cast(mat.alpha/6,'like',prototype).*S);
end
