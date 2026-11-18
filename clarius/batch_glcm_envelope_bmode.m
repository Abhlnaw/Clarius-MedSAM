function report = batch_glcm_envelope_bmode(manifestPath,outputPath,options)
% Application adapter for the student's supplied batch_glcm_envelope_bmode.m.
% Input JSON: jobs with id, raw, metadata_yml (one preserved RF frame each).
% All envelope formation, filtering, resampling and GLCM happen in MATLAB.
arguments
    manifestPath (1,:) char
    outputPath (1,:) char
    options.Levels (1,:) double = [8 16 32 64]
    options.WindowPulseLengths (1,:) double = [5 10 20]
    options.IsoSpacingMm (1,1) double {mustBeFinite,mustBeNonnegative} = .130
end
manifest=jsondecode(fileread(manifestPath));
assert(isfield(manifest,'jobs') && ~isempty(manifest.jobs),'No RF jobs supplied');
if ~isfolder(outputPath),mkdir(outputPath);end
summaries=cell(0,1); records=cell(0,1); failures=cell(0,1);
for k=1:numel(manifest.jobs)
    job=manifest.jobs(k);
    try
        [rf,z,x,source]=read_clarius_rf_physical(job.raw,job.metadata_yml);
        spacing=options.IsoSpacingMm;
        if spacing==0,spacing=max(source.axial_spacing_mm,source.lateral_spacing_mm);end
        result=clarius_glcm_physical(rf,z,x,'TransmitFrequencyHz',source.transmit_frequency_Hz, ...
            'IsoSpacingMm',spacing,'Levels',options.Levels,'WindowPulseLengths',options.WindowPulseLengths);
        result.source=source;
        result.source_provenance=job;
        file=fullfile(outputPath,[job.id '_physical_glcm.mat']);
        assert(~isfile(file),'Choose a fresh output folder; refusing to overwrite saved feature maps');
        save(file,'result','-v7.3');
        records{end+1,1}=struct('id',job.id,'file',file,'native_spacing_mm',result.resampling.native_spacing_mm, ...
            'isotropic_size',result.resampling.isotropic_size,'isotropic_spacing_mm',spacing);
        for c=1:numel(result.configurations)
            config=result.configurations{c};
            for d=1:4
                values=reshape(config.feature_maps(:,:,d,:),[],2);
                supported=all(isfinite(values),2); pairs=config.valid_pairs(:,:,d); occupancy=config.occupied_fraction(:,:,d);
                values=values(supported,:);
                item=struct('File',string(job.id),'Domain',string(config.domain),'Levels',config.levels, ...
                    'WindowPL',config.window_pulse_lengths,'RequestedWindowMm',config.requested_window_mm, ...
                    'ActualWindowMm',config.actual_window_mm,'WindowPixels',config.window_pixels(1), ...
                    'Direction',string(result.direction_names{d}),'OffsetDistanceMm',result.offset_distances_mm(d), ...
                    'ValidWindows',nnz(supported),'TotalWindows',numel(supported), ...
                    'MeanValidPairs',mean(pairs(supported),'omitnan'),'MeanOccupiedFraction',mean(occupancy(supported),'omitnan'), ...
                    'MeanContrast',mean(values(:,1),'omitnan'), ...
                    'MeanNormalizedContrast',mean(values(:,1),'omitnan')/(config.levels-1)^2, ...
                    'MeanEntropyBits',mean(values(:,2),'omitnan'), ...
                    'StdNormalizedContrast',std(values(:,1)/(config.levels-1)^2,0,'omitnan'), ...
                    'StdEntropyBits',std(values(:,2),0,'omitnan'));
                summaries{end+1,1}=item;
            end
        end
        fprintf('[%d/%d] %s: %.6f x %.6f mm -> %.3f mm isotropic\n',k,numel(manifest.jobs),job.id(1:3),source.axial_spacing_mm,source.lateral_spacing_mm,spacing);
    catch exception
        failures{end+1,1}=struct('id',job.id,'error',exception.message);
        fprintf(2,'FAILED %s: %s\n',job.id,exception.message);
    end
end
if ~isempty(summaries),writetable(struct2table(vertcat(summaries{:})),fullfile(outputPath,'summary.csv'));end
report=struct('implementation','physical_glcm_v1','processing_engine',['MATLAB ' version], ...
    'features',{{'Contrast','Entropy'}},'direction_aggregation','NONE','records',{records},'failures',{failures});
fid=fopen(fullfile(outputPath,'report.json'),'w');assert(fid~=-1,'Cannot write report');
cleanup=onCleanup(@() fclose(fid));fprintf(fid,'%s',jsonencode(report));clear cleanup
assert(isempty(failures),'One or more physical GLCM jobs failed; inspect report.json');
end
