function [features,diagnostics,counts] = clarius_glcm_direction(q,levels,offset)
% Exact masked symmetric GLCM. Features = [contrast, entropy in bits].
% NaNs exclude pixels; both endpoints of a pair must be valid in this patch.
arguments
    q (:,:) double
    levels (1,1) double {mustBeInteger,mustBeGreaterThanOrEqual(levels,2)}
    offset (1,2) double {mustBeInteger}
end
assert(any(offset~=0),'Offset cannot be zero');
assert(~any(isinf(q(:))),'Infinite quantized values are invalid');
finite=q(isfinite(q));
assert(all(finite>=1 & finite<=levels & finite==floor(finite)), 'Quantized values must be integer bins 1..levels or NaN');
[h,w]=size(q); dr=offset(1); dc=offset(2);
r=max(1,1-dr):min(h,h-dr); c=max(1,1-dc):min(w,w-dc);
counts=sparse(levels,levels); features=[NaN NaN];
nPairs=0;
if ~isempty(r) && ~isempty(c)
    a=q(r,c); b=q(r+dr,c+dc); valid=isfinite(a)&isfinite(b);
    nPairs=nnz(valid);
    if nPairs
        counts=sparse(a(valid),b(valid),ones(nPairs,1),levels,levels);
        counts=counts+counts.';
        [i,j,n]=find(counts); p=n/sum(n);
        features=[sum(p.*(i-j).^2), -sum(p.*log2(p))];
    end
end
diagnostics=struct('valid_pairs',nPairs,'symmetric_count_total',2*nPairs, ...
    'occupied_fraction',nnz(counts)/(levels*levels),'symmetric',true);
end
