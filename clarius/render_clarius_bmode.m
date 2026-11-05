function [displayed, toned] = render_clarius_bmode(env, outputSize, gammaValue)
% Fixed gamma on original Clarius uint8 B-mode samples, then full-frame resize.
% B-mode is already detected/log-compressed; do not Hilbert/log/min-max it.
arguments
    env (:,:) uint8 {mustBeNonempty}
    outputSize (1,2) double {mustBeInteger,mustBePositive} = [853 1418]
    gammaValue (1,1) double {mustBeFinite,mustBePositive} = 0.8
end
toned = uint8(floor(255 * (double(env) / 255).^gammaValue + 0.5));
displayed = imresize(toned, outputSize, 'bilinear', 'Antialiasing', true);
end
