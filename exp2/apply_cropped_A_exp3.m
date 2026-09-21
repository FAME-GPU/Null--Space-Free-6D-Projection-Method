function Afield = apply_cropped_A_exp3(field,h)
%APPLY_CROPPED_A_EXP3 Matrix-free cropped Yee curl-curl action.
    u1=field.u1; u2=field.u2; u3=field.u3;
    h2inv=(1/h)^2;

    Dxx1=Dxx(u1); Dyy1=Dyy(u1); Dzz1=Dzz(u1);
    Dxx2=Dxx(u2); Dyy2=Dyy(u2); Dzz2=Dzz(u2);
    Dxx3=Dxx(u3); Dyy3=Dyy(u3); Dzz3=Dzz(u3);

    Au1=h2inv*(Dyy1+Dzz1-DyT_DxP(u2)-DzT_DxP(u3));
    Au2=h2inv*(-DxT_DyP(u1)+Dxx2+Dzz2-DzT_DyP(u3));
    Au3=h2inv*(-DxT_DzP(u1)-DyT_DzP(u2)+Dxx3+Dyy3);
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
