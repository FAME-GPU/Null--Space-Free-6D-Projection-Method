function yee = build_yee_points_exp3(nCrop,h,prototype)
%BUILD_YEE_POINTS_EXP3 Build three staggered Omega_n^+ point sets on device.
    [r1,r2,r3,g]=build_yee_edge_points_plus(nCrop,h,prototype);
    yee=struct();
    yee.r1=r1; yee.r2=r2; yee.r3=r3;
    yee.szPlus=g.szPlus; yee.szInner=g.szInner;
    yee.innerRangeInPlus=g.innerRangeInPlus;
    yee.numPlus=g.numPlus; yee.h=h; yee.n=nCrop;
end
