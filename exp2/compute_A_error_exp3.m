function eA = compute_A_error_exp3(testField,refA,h)
%COMPUTE_A_ERROR_EXP3 Relative error after cropped Yee curl-curl action.
    At=apply_cropped_A_exp3(testField,h);
    d1=At.u1-refA.u1; d2=At.u2-refA.u2; d3=At.u3-refA.u3;
    num=sqrt(gather_scalar(sum(abs(d1(:)).^2)+sum(abs(d2(:)).^2)+sum(abs(d3(:)).^2)));
    den=sqrt(gather_scalar(sum(abs(refA.u1(:)).^2)+sum(abs(refA.u2(:)).^2)+sum(abs(refA.u3(:)).^2)));
    eA=num/max(den,eps);
end
