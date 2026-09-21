function y = applyMij_rank1_exact(x,kernelFFT,plan)
%APPLYMIJ_RANK1_EXACT Apply one scalar Fourier multiplication block exactly.

    X=rank1_exact_fft_scatter(x,plan);
    y=rank1_exact_ifft_gather(kernelFFT.*X,plan);
end
