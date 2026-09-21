function epsField = sample_scalar_yee_material_exp4(cfg,st,h,n,prototype)
%SAMPLE_SCALAR_YEE_MATERIAL_EXP4 Evaluate epsilon on Yee edges on the GPU.
    if nargin<5 || isempty(prototype), prototype=st.P(1); end
    if ~isa(prototype,'gpuArray')
        error('Yee material sampling must run on the GPU.');
    end

    idx=cast((-n:n).*h,'like',prototype);
    epsField=struct();
    epsField.u1=eval_component(idx+h/2,idx,idx,st.P,cfg.material,st.materialRuntime);
    epsField.u2=eval_component(idx,idx+h/2,idx,st.P,cfg.material,st.materialRuntime);
    epsField.u3=eval_component(idx,idx,idx+h/2,st.P,cfg.material,st.materialRuntime);
end

function e=eval_component(x,y,z,P,mat,runtime)
    X=reshape(x,[],1,1); Y=reshape(y,1,[],1); Z=reshape(z,1,1,[]);

    p1=P(1,1).*X+P(1,2).*Y+P(1,3).*Z;
    p2=P(2,1).*X+P(2,2).*Y+P(2,3).*Z;
    p3=P(3,1).*X+P(3,2).*Y+P(3,3).*Z;
    p4=P(4,1).*X+P(4,2).*Y+P(4,3).*Z;
    p5=P(5,1).*X+P(5,2).*Y+P(5,3).*Z;
    p6=P(6,1).*X+P(6,2).*Y+P(6,3).*Z;

    e=evaluate_exp4_permittivity(mat,p1,p2,p3,p4,p5,p6);
end
