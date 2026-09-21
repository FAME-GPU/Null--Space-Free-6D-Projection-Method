function small = crop_plus_field_exp4(large,nLarge,nSmall)
%CROP_PLUS_FIELD_EXP4 Crop a same-h Omega_n^+ field about the origin.
    if nSmall>nLarge, error('nSmall must not exceed nLarge.'); end
    c=nLarge+2; % index of physical grid index zero in -(nLarge+1):(nLarge+1)
    r=(c-(nSmall+1)):(c+(nSmall+1));
    small=struct('u1',large.u1(r,r,r),'u2',large.u2(r,r,r),'u3',large.u3(r,r,r));
end
