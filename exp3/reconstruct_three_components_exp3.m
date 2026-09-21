function [field,info] = reconstruct_three_components_exp3(X,qModes,yeePts,method,opt)
%RECONSTRUCT_THREE_COMPONENTS_EXP3 Reconstruct all staggered components.
    NF=size(qModes,1);
    if numel(X)~=3*NF, error('Fourier eigenvector must contain 3*NF entries.'); end
    coeff={X(1:NF),X(NF+1:2*NF),X(2*NF+1:3*NF)};
    points={yeePts.r1,yeePts.r2,yeePts.r3};
    u=cell(3,1); infoc=cell(3,1);

    wait_for_gpu(); t0=tic;
    for ell=1:3
        switch lower(method)
            case 'direct'
                [u{ell},infoc{ell}]=reconstruct_field_direct_blocked_exp3( ...
                    coeff{ell},qModes,points{ell},opt.blockSize);
            case 'single'
                [u{ell},infoc{ell}]=reconstruct_field_taylor_single_exp3( ...
                    coeff{ell},qModes,points{ell},opt.order,opt.termBatchSize,opt.center);
            case 'multi'
                [u{ell},infoc{ell}]=reconstruct_field_taylor_multi_exp3( ...
                    coeff{ell},qModes,points{ell},opt.order,opt.termBatchSize,opt.blocks);
            otherwise
                error('Unknown reconstruction method: %s',method);
        end
    end
    wait_for_gpu(u{3}); total=toc(t0);

    field=struct();
    field.u1=reshape(u{1},yeePts.szPlus);
    field.u2=reshape(u{2},yeePts.szPlus);
    field.u3=reshape(u{3},yeePts.szPlus);
    info=struct();
    info.method=method;
    info.componentInfo=infoc;
    info.timeSeconds=total;
    info.estimatedPeakAuxBytes=max(cellfun(@(s)s.estimatedPeakAuxBytes,infoc));
    if isfield(infoc{1},'rhoMax')
        info.rhoMax=max(cellfun(@(s)s.rhoMax,infoc));
        info.numTerms=infoc{1}.numTerms;
    else
        info.rhoMax=NaN; info.numTerms=NaN;
    end
end
