function result = clarius_glcm_physical(samples,zNative,xNative,options)
% Calibrated two-feature GLCM, adapted from the supplied PhD-student script.
% RF -> linear envelope -> isotropic grid -> linear/log quantization -> maps.
% No clustering, direction averaging, segmentation or per-window rescaling.
arguments
    samples (:,:) double {mustBeFinite}
    zNative (:,1) double {mustBeFinite}
    xNative (1,:) double {mustBeFinite}
    options.InputDomain (1,1) string {mustBeMember(options.InputDomain,["RF","Envelope"])} = "RF"
    options.IsoSpacingMm (1,1) double {mustBeFinite,mustBePositive} = .130
    options.Levels (1,:) double {mustBeInteger,mustBeGreaterThanOrEqual(options.Levels,2)} = [8 16 32 64]
    options.WindowPulseLengths (1,:) double {mustBeFinite,mustBePositive} = [5 10 20]
    options.TransmitFrequencyHz (1,1) double {mustBeFinite,mustBePositive}
    options.SoundSpeedMps (1,1) double {mustBeFinite,mustBePositive} = 1540
    options.CyclesPerPulse (1,1) double {mustBeFinite,mustBePositive} = 2
    options.OverlapFraction (1,1) double {mustBeFinite,mustBeNonnegative,mustBeLessThan(options.OverlapFraction,1)} = .80
    options.DynamicRangeDb (1,1) double {mustBeFinite,mustBePositive} = 60
    options.MinValidFraction (1,1) double {mustBeFinite,mustBePositive,mustBeLessThanOrEqual(options.MinValidFraction,1)} = .50
    options.RoiMask logical = true(size(samples))
end
assert(isequal(size(options.RoiMask),size(samples)),'ROI must be on the native source grid');
if options.InputDomain=="RF", envelope=abs(hilbert(samples));
else, assert(all(samples(:)>=0),'Linear envelope amplitudes cannot be negative'); envelope=samples; end
[envIso,zIso,xIso,resampling]=clarius_isotropic_envelope(envelope,zNative,xNative,options.IsoSpacingMm);
[Z,X]=ndgrid(zIso,xIso);
roiIso=interp2(xNative,zNative,double(options.RoiMask),X,Z,'nearest',0)>.5;
valid=isfinite(envIso)&envIso>0; % Matches supplied script: zero amplitude is excluded.
assert(any(valid(:)),'No positive finite envelope samples');
vMin=min(envIso(valid)); vMax=max(envIso(valid));
bmodeDb=nan(size(envIso)); referenceAmplitude=vMax;
bmodeDb(valid)=max(-options.DynamicRangeDb,min(0,20*log10(envIso(valid)/referenceAmplitude)));
offsets=[0 1;-1 0;-1 1;-1 -1];
directions={'Horizontal','Axial','DiagonalUpRight','DiagonalUpLeft'};
wavelengthMm=options.SoundSpeedMps/options.TransmitFrequencyHz*1e3;
pulseLengthMm=options.CyclesPerPulse*wavelengthMm;
result=struct('implementation','physical_glcm_v1','feature_names',{{'Contrast','Entropy'}}, ...
    'direction_names',{directions},'direction_aggregation','NONE','offsets_row_col',offsets, ...
    'offset_distances_mm',sqrt(sum(offsets.^2,2))*options.IsoSpacingMm, ...
    'resampling',resampling,'zIso_mm',zIso,'xIso_mm',xIso,'envelope_isotropic',envIso, ...
    'bmode_dB',bmodeDb,'roi_isotropic',roiIso,'input_domain',char(options.InputDomain), ...
    'quantization_scope','Positive finite whole-image envelope, before ROI masking; never requantize windows', ...
    'envelope_min',vMin,'envelope_max',vMax,'bmode_reference_amplitude',referenceAmplitude, ...
    'bmode_dynamic_range_dB',options.DynamicRangeDb,'cycles_per_pulse_assumed',options.CyclesPerPulse, ...
    'transmit_frequency_Hz',options.TransmitFrequencyHz,'wavelength_mm',wavelengthMm, ...
    'pulse_length_mm',pulseLengthMm,'inverse_TGC_applied',false,'options',options);
configurations=cell(0,1);
for levels=options.Levels
    qEnvelope=nan(size(envIso));
    if vMax>vMin, qEnvelope(valid)=min(levels,1+floor((envIso(valid)-vMin)/(vMax-vMin)*levels));
    else, qEnvelope(valid)=1; end
    qBmode=nan(size(envIso));
    qBmode(valid)=min(levels,1+floor((bmodeDb(valid)+options.DynamicRangeDb)/options.DynamicRangeDb*levels));
    quantized={qEnvelope,qBmode}; domains={'Envelope','Bmode'};
    for w=options.WindowPulseLengths
        desiredMm=w*pulseLengthMm;
        win=max(3,2*round((desiredMm/options.IsoSpacingMm-1)/2)+1);
        half=floor(win/2); targetStep=max(1,win*(1-options.OverlapFraction));
        rows=centers(half+1,size(envIso,1)-half,targetStep);
        cols=centers(half+1,size(envIso,2)-half,targetStep);
        assert(~isempty(rows) && ~isempty(cols),'Requested physical window exceeds image size');
        for domain=1:2
            q=quantized{domain}; q(~roiIso)=NaN;
            maps=nan(numel(rows),numel(cols),4,2);
            pairs=zeros(numel(rows),numel(cols),4);
            occupancy=nan(size(pairs));
            for ir=1:numel(rows)
                rr=rows(ir)-half:rows(ir)+half;
                for ic=1:numel(cols)
                    if ~roiIso(rows(ir),cols(ic)) || ~isfinite(q(rows(ir),cols(ic))),continue;end
                    cc=cols(ic)-half:cols(ic)+half; patch=q(rr,cc);
                    if nnz(isfinite(patch))<ceil(options.MinValidFraction*win^2),continue;end
                    for d=1:4
                        [f,audit]=clarius_glcm_direction(patch,levels,offsets(d,:));
                        maps(ir,ic,d,:)=reshape(f,[1 1 1 2]);
                        pairs(ir,ic,d)=audit.valid_pairs;
                        occupancy(ir,ic,d)=audit.occupied_fraction;
                    end
                end
            end
            config=struct('domain',domains{domain},'levels',levels,'window_pulse_lengths',w, ...
                'requested_window_mm',desiredMm,'actual_window_mm',win*options.IsoSpacingMm, ...
                'window_pixels',[win win],'requested_overlap',options.OverlapFraction, ...
                'actual_overlap_row',overlap(win,rows),'actual_overlap_col',overlap(win,cols), ...
                'row_centers',rows,'col_centers',cols,'z_centers_mm',zIso(rows),'x_centers_mm',xIso(cols), ...
                'quantized_image',q,'feature_maps',maps,'valid_pairs',pairs,'occupied_fraction',occupancy);
            configurations{end+1,1}=config;
        end
    end
end
result.configurations=configurations;
end

function output=centers(first,last,target)
output=[]; if first>last,return;end
output=first; base=floor(target); fraction=target-base; accumulator=0;
while true
    accumulator=accumulator+fraction; step=base;
    if accumulator>=1-1e-12,step=step+1;accumulator=accumulator-1;end
    next=output(end)+step; if next>last,break;end
    output(end+1)=next;
end
end

function value=overlap(window,positions)
if numel(positions)<2,value=NaN;else,value=mean((window-diff(positions))/window);end
end
