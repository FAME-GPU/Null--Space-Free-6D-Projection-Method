function Afield = apply_cropped_A_exp3(field,h)
%APPLY_CROPPED_A_EXP3 Matrix-free cropped Yee curl-curl action.
%
% The three output components are assembled sequentially to keep the large
% n=160 GPU workspace bounded.  This is algebraically identical to the
% previous block formula but avoids retaining nine second-difference arrays.

    u1=field.u1; u2=field.u2; u3=field.u3;
    h2inv=(1/h)^2;

    Au1=Dyy(u1);
    t=Dzz(u1); Au1=Au1+t; clear t
    t=DyT_DxP(u2); Au1=Au1-t; clear t
    t=DzT_DxP(u3); Au1=Au1-t; clear t
    Au1=h2inv*Au1;

    Au2=Dxx(u2);
    t=Dzz(u2); Au2=Au2+t; clear t
    t=DxT_DyP(u1); Au2=Au2-t; clear t
    t=DzT_DyP(u3); Au2=Au2-t; clear t
    Au2=h2inv*Au2;

    Au3=Dxx(u3);
    t=Dyy(u3); Au3=Au3+t; clear t
    t=DxT_DzP(u1); Au3=Au3-t; clear t
    t=DyT_DzP(u2); Au3=Au3-t; clear t
    Au3=h2inv*Au3;

    Afield=struct('u1',Au1,'u2',Au2,'u3',Au3);
end

function y=Dxx(u)
    y=-u(1:end-2,2:end-1,2:end-1)+2*u(2:end-1,2:end-1,2:end-1)-u(3:end,2:end-1,2:end-1);
end
function y=Dyy(u)
    y=-u(2:end-1,1:end-2,2:end-1)+2*u(2:end-1,2:end-1,2:end-1)-u(2:end-1,3:end,2:end-1);
end
function y=Dzz(u)
    y=-u(2:end-1,2:end-1,1:end-2)+2*u(2:end-1,2:end-1,2:end-1)-u(2:end-1,2:end-1,3:end);
end
function y=DyT_DxP(u)
    y=u(3:end,1:end-2,2:end-1)-u(2:end-1,1:end-2,2:end-1)-u(3:end,2:end-1,2:end-1)+u(2:end-1,2:end-1,2:end-1);
end
function y=DzT_DxP(u)
    y=u(3:end,2:end-1,1:end-2)-u(2:end-1,2:end-1,1:end-2)-u(3:end,2:end-1,2:end-1)+u(2:end-1,2:end-1,2:end-1);
end
function y=DxT_DyP(u)
    y=-u(1:end-2,2:end-1,2:end-1)+u(2:end-1,2:end-1,2:end-1)+u(1:end-2,3:end,2:end-1)-u(2:end-1,3:end,2:end-1);
end
function y=DzT_DyP(u)
    y=-u(2:end-1,2:end-1,1:end-2)+u(2:end-1,3:end,1:end-2)+u(2:end-1,2:end-1,2:end-1)-u(2:end-1,3:end,2:end-1);
end
function y=DxT_DzP(u)
    y=-u(1:end-2,2:end-1,2:end-1)+u(2:end-1,2:end-1,2:end-1)+u(1:end-2,2:end-1,3:end)-u(2:end-1,2:end-1,3:end);
end
function y=DyT_DzP(u)
    y=-u(2:end-1,1:end-2,2:end-1)+u(2:end-1,2:end-1,2:end-1)+u(2:end-1,1:end-2,3:end)-u(2:end-1,2:end-1,3:end);
end
