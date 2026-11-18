function [out,zIso,xIso,audit] = clarius_isotropic_envelope(envelope,z,x,spacing)
% Adapted from batch_glcm_envelope_bmode.m supplied by the user's PhD student.
% Uniform Cartesian axes only. Anti-alias BEFORE interpolation/downsampling.
arguments
    envelope (:,:) double {mustBeFinite,mustBeNonnegative}
    z (:,1) double {mustBeFinite}
    x (1,:) double {mustBeFinite}
    spacing (1,1) double {mustBeFinite,mustBePositive}
end
assert(isequal(size(envelope),[numel(z) numel(x)]),'Axes and samples differ');
assert(numel(z)>=3 && numel(x)>=3,'Need at least three samples per axis');
dz=median(diff(z)); dx=median(diff(x));
assert(dz>0 && dx>0 && all(abs(diff(z)-dz)<max(1e-9,dz*1e-6)) && ...
    all(abs(diff(x)-dx)<max(1e-9,dx*1e-6)), 'Axes must increase uniformly; scan-convert non-Cartesian data first');
assert(spacing>=max(dz,dx)-1e-10,'Isotropic grid must not upsample either axis');
A=envelope; orders=[0 0];
if spacing>dz*1.001, [A,orders(1)]=antialias(A,1,spacing/dz); end
if spacing>dx*1.001, [A,orders(2)]=antialias(A,2,spacing/dx); end
zIso=z(1):spacing:z(end); xIso=x(1):spacing:x(end);
assert(numel(zIso)>=3 && numel(xIso)>=3,'Isotropic grid is too small');
[Z,X]=ndgrid(zIso,xIso);
out=interp2(x,z,A,X,Z,'linear');
out=max(out,0); % Small FIR overshoots may be negative.
audit=struct('native_spacing_mm',[dz dx],'isotropic_spacing_mm',[spacing spacing], ...
    'native_size',size(envelope),'isotropic_size',size(out), ...
    'antialias_filter_order',orders,'antialias_cutoff','0.90 / downsampling ratio, capped at 0.99 Nyquist', ...
    'interpolation','linear on physical coordinates','speckle_filtering',false, ...
    'last_center_remainder_mm',[z(end)-zIso(end) x(end)-xIso(end)]);
end

function [out,order]=antialias(A,dimension,ratio)
n=size(A,dimension);
order=min(48,2*floor((n-4)/6));
assert(order>=8,'Insufficient samples for the anti-alias filter');
b=fir1(order,min(.99,.90/ratio));
if dimension==1, out=filtfilt(b,1,A); else, out=filtfilt(b,1,A.').'; end
end
