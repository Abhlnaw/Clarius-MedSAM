function report = export_clarius_bmode(manifestPath, outputPath, comparisonPath)
% Read byte-preserved ENV frames and perform all image processing in MATLAB.
% comparisonPath optionally contains earlier Python PNGs for numerical audit.
arguments
    manifestPath (1,:) char
    outputPath (1,:) char
    comparisonPath (1,:) char = ''
end
manifest = jsondecode(fileread(manifestPath));
assert(isempty(manifest.failures), 'Prepared frame manifest contains failures');
assert(numel(manifest.jobs) == manifest.expected_captures, 'Incomplete capture set');
imagePath = fullfile(outputPath, 'images');
if ~isfolder(imagePath), mkdir(imagePath); end
records = cell(numel(manifest.jobs), 1);
for index = 1:numel(manifest.jobs)
    job = manifest.jobs(index);
    fid = fopen(job.raw, 'r', 'ieee-le');
    assert(fid ~= -1, 'Cannot open prepared B-mode frame');
    cleanup = onCleanup(@() fclose(fid));
    header = fread(fid, 5, '*uint32');
    assert(numel(header) == 5 && header(1) == 1 && header(2) == 1 && header(5) == 1, ...
        'Expected a one-frame uint8 Clarius ENV stream');
    timestamp = fread(fid, 1, '*uint64');
    assert(numel(timestamp) == 1, 'Missing frame timestamp');
    [env, count] = fread(fid, [double(header(4)), double(header(3))], '*uint8');
    assert(count == double(header(4))*double(header(3)), 'Truncated B-mode samples');
    assert(isempty(fread(fid, 1, '*uint8')), 'Unexpected extra frame bytes');
    assert(strcmp(sprintf('%u', timestamp), job.timestamp_raw), 'Timestamp mismatch');
    clear cleanup
    [displayed, toned] = render_clarius_bmode(env);
    name = [job.id '.png'];
    destination = fullfile(imagePath, name);
    if isfile(destination)
        assert(isequal(imread(destination), displayed), 'Refusing to overwrite differing PNG');
    else
        imwrite(displayed, destination);
    end
    item = struct('id', job.id, 'source_archive', job.source_archive, ...
        'source_member', job.source_member, 'raw_sha256', job.raw_sha256, ...
        'selected_frame', job.selected_frame, 'timestamp_raw', job.timestamp_raw, ...
        'timestamp_delta_ns', job.timestamp_delta_ns, 'native_shape_hw', size(env), ...
        'output_shape_hw', size(displayed), 'gamma', 0.8, ...
        'gamma_formula', 'floor(255 * (double(env)/255).^0.8 + 0.5)', ...
        'processing_engine', ['MATLAB ' version], ...
        'resize_method', 'MATLAB imresize bilinear, Antialiasing true, full frame', ...
        'cropped', false, 'native_black_pixels', nnz(env == 0), ...
        'toned_black_pixels', nnz(toned == 0), 'path', destination);
    if ~isempty(comparisonPath)
        prior = imread(fullfile(comparisonPath, name));
        assert(isequal(size(prior),size(displayed)), 'Prior image grid differs');
        difference = abs(double(displayed)-double(prior));
        item.comparison = struct('mean_absolute_gray_error', mean(difference(:)), ...
            'max_absolute_gray_error', max(difference(:)), ...
            'pixel_equal_fraction', mean(difference(:)==0), ...
            'image_ssim', ssim(displayed,prior), ...
            'note', 'Rendering agreement only; not segmentation accuracy');
    end
    sidecar = fopen(fullfile(imagePath,[job.id '.json']), 'w');
    assert(sidecar ~= -1, 'Cannot create image metadata');
    fprintf(sidecar, '%s', jsonencode(item)); fclose(sidecar);
    records{index} = item;
    fprintf('MATLAB exported %d/%d: %s\n',index,numel(records),job.id(1:3));
end
report = struct('processing_engine', ['MATLAB ' version], ...
    'python_role', 'Archive extraction, LZO decompression, binary frame selection only', ...
    'variant', 'env_gamma08', 'output_size_hw', [853 1418], ...
    'cropped', false, 'cases', {records}, 'failures', []);
handle = fopen(fullfile(outputPath,'manifest.json'), 'w');
assert(handle ~= -1, 'Cannot create output manifest');
fprintf(handle, '%s', jsonencode(report)); fclose(handle);
end
