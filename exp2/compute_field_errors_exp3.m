function out = compute_field_errors_exp3(testField,refField)
%COMPUTE_FIELD_ERRORS_EXP3 Three-component relative 2- and infinity-errors.
    d1=testField.u1-refField.u1;
    d2=testField.u2-refField.u2;
    d3=testField.u3-refField.u3;
    num2=sqrt(gather_scalar(sum(abs(d1(:)).^2)+sum(abs(d2(:)).^2)+sum(abs(d3(:)).^2)));
    den2=sqrt(gather_scalar(sum(abs(refField.u1(:)).^2)+sum(abs(refField.u2(:)).^2)+sum(abs(refField.u3(:)).^2)));
    numInf=gather_scalar(max([max(abs(d1(:))),max(abs(d2(:))),max(abs(d3(:)))]));
    denInf=gather_scalar(max([max(abs(refField.u1(:))),max(abs(refField.u2(:))),max(abs(refField.u3(:)))]));
    out=struct('e2',num2/max(den2,eps),'eInf',numInf/max(denInf,eps));
end
