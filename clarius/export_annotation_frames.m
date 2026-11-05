function export_annotation_frames(manifestPath, outputPath)
% Export first frames using the same RF B-mode calculation as runme2_export.
manifest = jsondecode(fileread(manifestPath));
imagePath = fullfile(outputPath, 'images');
mkdir(imagePath);
for i = 1:numel(manifest.jobs)
    job = manifest.jobs(i);
    fid = fopen(job.raw, 'r', 'ieee-le');
    assert(fid ~= -1, 'Cannot open RF frame');
    cleanup = onCleanup(@() fclose(fid));
    header = fread(fid, 5, 'int32');
    fread(fid, 1, 'int64');
    RF = fread(fid, [header(4), header(3)], 'int16');
    assert(isequal(size(RF), [header(4), header(3)]), 'Incomplete RF frame');
    BB = 20*log10(1 + abs(hilbert(RF)))-100;
    limits = [min(BB(:)), max(BB(:))];
    pixels = uint8(round(255 * (BB-limits(1)) / max(diff(limits), eps)));
    % Resize the entire RF frame; no cropping. Annotation and model inputs
    % share this grid. This display aspect is not physical calibration.
    exported = imresize(pixels, [853 1418], 'bilinear');
    imwrite(exported, fullfile(imagePath, [job.id '.png']));
    metadata = struct('source_archive', job.source_archive, ...
        'source_member', job.source_member, 'frame', 1, ...
        'height', size(exported,1), 'width', size(exported,2), ...
        'native_height', size(pixels,1), 'native_width', size(pixels,2), ...
        'resize_method', 'MATLAB imresize bilinear, full frame, no crop', ...
        'intensity_limits', limits, 'timestamp_raw', job.timestamp_raw, ...
        'note', 'Full-frame RF-derived B-mode resized to 1418 x 853 for annotation and model evaluation; aspect is not physically calibrated.');
    meta = fopen(fullfile(imagePath, [job.id '.json']), 'w');
    assert(meta ~= -1, 'Cannot write metadata');
    fprintf(meta, '%s', jsonencode(metadata)); fclose(meta);
    clear cleanup
    fprintf('Exported %d/%d: %s\n', i, numel(manifest.jobs), job.id);
end
end
