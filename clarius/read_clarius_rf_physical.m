function [rf,z,x,metadata] = read_clarius_rf_physical(rawPath,yamlPath,frame,soundSpeed)
% Read an uncompressed Cartesian linear-array RF frame and its calibration.
% Axial coordinate is relative to the first stored sample; no absolute-depth
% registration to ENV/rendered PNGs is assumed. Steering/curved grids fail.
arguments
    rawPath (1,:) char
    yamlPath (1,:) char
    frame (1,1) double {mustBeInteger,mustBePositive} = 1
    soundSpeed (1,1) double {mustBeFinite,mustBePositive} = 1540
end
text=fileread(yamlPath);
fs=frequency(text,'sampling rate'); fc=frequency(text,'transmit frequency');
pitch=number(text,'pitch')/1000; % Clarius probe pitch is in micrometers.
assert(isfinite(pitch)&&pitch>0,'Missing positive probe pitch in micrometers');
radius=regexp(text,'(?m)^\s*radius:\s*([-+\d.eE]+)','tokens','once');
assert(isempty(radius)||str2double(radius{1})==0,'Curved probe requires scan conversion before Cartesian resampling');
fid=fopen(rawPath,'r','ieee-le'); assert(fid~=-1,'Cannot open RF stream');
cleanup=onCleanup(@() fclose(fid));
header=fread(fid,5,'*uint32');
assert(numel(header)==5 && header(5)==2,'Expected 16-bit beamformed RF samples');
assert(frame<=double(header(2)),'Requested RF frame does not exist');
h=double(header(4)); w=double(header(3));
assert(h>=3 && w>=3,'RF grid is too small');
dimensions=regexp(text,'size:\s*\{\s*samples per line:\s*(\d+),\s*number of lines:\s*(\d+),\s*sample size:\s*(\d+)','tokens','once');
assert(~isempty(dimensions)&&isequal(cellfun(@str2double,dimensions),[h w 2]), ...
    'RF YAML dimensions/sample type differ from RAW header');
status=fseek(fid,20+(frame-1)*(8+2*h*w),'bof'); assert(status==0,'RF frame seek failed');
timestamp=fread(fid,1,'*uint64'); [rf,count]=fread(fid,[h w],'*int16');
assert(~isempty(timestamp) && count==h*w,'Truncated RF frame'); rf=double(rf);
tokens=regexp(text,'\{\s*rx element:\s*([-+\d.eE]+),\s*tx element:\s*([-+\d.eE]+),\s*angle:\s*([-+\d.eE]+)','tokens');
assert(numel(tokens)==w,'Metadata must supply geometry for every RF line');
rx=cellfun(@(t) str2double(t{1}),tokens);
angles=cellfun(@(t) str2double(t{3}),tokens);
assert(all(isfinite(rx)) && all(isfinite(angles)) && all(abs(angles)<1e-9), ...
    'Steered RF needs explicit scan conversion; refusing a Cartesian-spacing assumption');
x=rx*pitch;
dx=median(diff(x));
assert(dx>0 && all(abs(diff(x)-dx)<1e-8),'RF receive-line positions must be uniformly increasing');
dz=soundSpeed/(2*fs)*1000; z=(0:h-1)'*dz;
metadata=struct('raw',rawPath,'yaml',yamlPath,'frame',frame, ...
    'timestamp_raw',sprintf('%u',timestamp),'native_size',[h w], ...
    'sample_rate_Hz',fs,'transmit_frequency_Hz',fc,'sound_speed_mps',soundSpeed, ...
    'probe_pitch_mm',pitch,'axial_spacing_mm',dz,'lateral_spacing_mm',dx, ...
    'geometry','Uniform zero-steering receive-element positions times probe pitch', ...
    'depth_origin','Relative to first stored RF sample; absolute delay not registered', ...
    'inverse_TGC_applied',false);
delay=regexp(text,'(?m)^delay samples:\s*(\d+)','tokens','once');
if ~isempty(delay),metadata.reported_delay_samples=str2double(delay{1});end
end

function value=frequency(text,key)
pattern=['(?m)^' regexptranslate('escape',key) ':\s*([-+\d.eE]+)\s*(MHz|kHz|Hz)'];
token=regexp(text,pattern,'tokens','once'); assert(~isempty(token),['Missing ' key]);
value=str2double(token{1});
if strcmp(token{2},'MHz'),value=value*1e6;elseif strcmp(token{2},'kHz'),value=value*1e3;end
assert(isfinite(value)&&value>0,['Invalid ' key]);
end

function value=number(text,key)
token=regexp(text,['(?m)^\s*' regexptranslate('escape',key) ':\s*([-+\d.eE]+)'],'tokens','once');
assert(~isempty(token),['Missing ' key]); value=str2double(token{1});
end
