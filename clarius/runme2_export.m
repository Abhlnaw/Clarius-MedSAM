% Combined script to read and display Clarius research data (RF/IQ/B)
% Author: Reza Zahiri / Updated for Auto-Detection & Python LZO

close all
clear all
clc
addpath(fileparts(mfilename('fullpath')));

% Dock figures as tabs in a single window
set(0, 'DefaultFigureWindowStyle', 'docked');

% Set overall exam directory path
path = 'C:\Users\abhit\Documents\Hoyt Research\data\exam05'; % change this to current exam
exportLegacyRF = false; % optional older RF export; preferred export uses ENV below

%% 0. Optional: Run Python conversion/decompression only if needed
rf_files = dir(fullfile(path, '*_rf.raw'));
env_files = dir(fullfile(path, '*_env.raw'));
lzo_files = dir(fullfile(path, '*.lzo'));

if isempty(rf_files) || isempty(env_files)
    if ~isempty(lzo_files)
        pyScript = fullfile('C:\Users\abhit\Documents\Hoyt Research', 'clarius_convert.py');

        if isfile(pyScript)
            fprintf('Raw files missing; running Clarius converter on: %s\n', path);
            cmd = sprintf('python "%s" "%s"', pyScript, path);
            status = system(cmd);
            if status ~= 0
                error('Python conversion failed.');
            end
        else
            warning('Converter script not found: %s', pyScript);
        end
    else
        warning('No decompressed raw files found, and no .lzo files to convert.');
    end
else
    fprintf('Decompressed raw files already exist; skipping conversion.\n');
end

%% Probe Metadata (.yml reference)
depth_mm = 18.70;
pitch_mm = 0.130;

%% 1. Read RF data (Auto-detect *_rf.raw)
rf_files = dir(fullfile(path, '*_rf.raw'));
if ~isempty(rf_files)
    filenameRF = rf_files(1).name; % Picks the first matching RF file in folder
    fullPathRF = fullfile(path, filenameRF);
    fprintf('Found RF file: %s\n', filenameRF);
    
    numFrames = 50;
    [dataRF, header, tsRF] = rdataread(fullPathRF, numFrames);
    if (numFrames > header.frames)
        numFrames = header.frames;
    end
    
    num_lines = header.lines;
    width_mm = (num_lines - 1) * pitch_mm;
    x_axis = linspace(-width_mm/2, width_mm/2, num_lines);
    z_axis_rf = linspace(0, depth_mm, header.samples);
    
    % --- CREATE FOLDER FOR PNG EXPORTS ---
    imgFolder = fullfile(path, 'exported_images_rf_legacy');
    if exportLegacyRF && ~exist(imgFolder, 'dir')
        mkdir(imgFolder);
    end
    
    figure('Name', 'Clarius RF Data');
    colormap(gray)
    
    for i = 1:numFrames
        RF = squeeze(dataRF(i,:,:));
        BB = 20*log10(1 + abs(hilbert(RF)))-100;
        
        subplot(1,2,1);
        imagesc(x_axis, z_axis_rf, RF, [-1000 1000]);
        title(sprintf('Raw RF (Frame %d)', i));
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        
        subplot(1,2,2);
        imagesc(x_axis, z_axis_rf, BB);
        title('Sample B from RF');
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        drawnow;
        
        % Export underlying B-mode pixels, without axes or screen scaling.
        % Match imagesc's per-frame display range; avoid the unused fixed
        % 15..70 range, which may clip the RF-derived values to black.
        if exportLegacyRF
        limits = [min(BB(:)), max(BB(:))];
        BB_norm = (BB - limits(1)) / max(limits(2) - limits(1), eps);
        imgName = sprintf('frame_%03d.png', i);
        imwrite(uint8(round(255 * BB_norm)), fullfile(imgFolder, imgName));
        metadata = struct('frame', i, 'source', fullPathRF, ...
            'intensity_limits', limits, 'height', size(BB, 1), ...
            'width', size(BB, 2), 'timestamp_raw', sprintf('%d', tsRF(i)), ...
            'note', 'Unscanconverted RF-derived B-mode; intensity is display-scaled, not calibrated.');
        fidMetadata = fopen(fullfile(imgFolder, sprintf('frame_%03d.json', i)), 'w');
        if fidMetadata == -1
            error('Could not create frame metadata file.');
        end
        fprintf(fidMetadata, '%s', jsonencode(metadata));
        fclose(fidMetadata);
        end

    end
    if exportLegacyRF
        fprintf('Saved %d legacy RF images to: %s\n', numFrames, imgFolder);
    end
else
    warning('No *_rf.raw file found in path: %s', path);
end

%% 2. Read B / Envelope data (Auto-detect *_env.raw)
env_files = dir(fullfile(path, '*_env.raw'));
if ~isempty(env_files)
    filenameB = env_files(1).name; % Picks the first matching ENV file in folder
    fullPathB = fullfile(path, filenameB);
    fprintf('Found Envelope file: %s\n', filenameB);
    
    numFrames = 50;
    [dataB, header, tsB] = rdataread(fullPathB, numFrames);
    if (numFrames > header.frames)
        numFrames = header.frames;
    end
    
    num_lines = header.lines;
    width_mm = (num_lines - 1) * pitch_mm / 2;
    x_axis = linspace(-width_mm/2, width_mm/2, num_lines);
    z_axis_env = linspace(0, depth_mm, header.samples);
    
    figure('Name', 'Clarius Envelope Data');
    colormap(gray)
    envFolder = fullfile(path, 'exported_images_env_gamma08');
    if ~isfolder(envFolder), mkdir(envFolder); end
    
    for i = 1:numFrames
        % All contrast, full-frame resizing, and PNG export happen in MATLAB.
        % ENV is already 8-bit B-mode: do not Hilbert/log/min-max it again.
        BB = uint8(squeeze(dataB(i,:,:)));
        [exported, toned] = render_clarius_bmode(BB);
        imagesc(x_axis, z_axis_env, toned, [0 255]);
        title(sprintf('B Pre-scan, gamma 0.8 (Frame %d)', i));
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        drawnow;
        destination = fullfile(envFolder, sprintf('frame_%03d.png', i));
        if isfile(destination)
            assert(isequal(imread(destination),exported), ...
                'Refusing to overwrite a differing ENV export: %s',destination);
        else
            imwrite(exported,destination);
        end
        metadata = struct('frame',i,'source',fullPathB,'gamma',0.8, ...
            'native_shape_hw',size(BB),'output_shape_hw',size(exported), ...
            'processing_engine',['MATLAB ' version],'cropped',false, ...
            'resize_method','MATLAB imresize bilinear, Antialiasing true, full frame', ...
            'timestamp_raw',sprintf('%u',tsB(i)));
        fidMetadata = fopen(fullfile(envFolder,sprintf('frame_%03d.json',i)),'w');
        assert(fidMetadata ~= -1,'Cannot create ENV metadata');
        fprintf(fidMetadata,'%s',jsonencode(metadata));fclose(fidMetadata);
    end
    fprintf('Saved %d MATLAB env_gamma08 images to: %s\n',numFrames,envFolder);
else
    warning('No *_env.raw file found in path: %s', path);
end

%% 3. Read IQ data (Auto-detect *_iq.raw)
iq_files = dir(fullfile(path, '*_iq.raw'));
if ~isempty(iq_files)
    filenameIQ = iq_files(1).name; % Picks the first matching IQ file in folder
    fullPathIQ = fullfile(path, filenameIQ);
    fprintf('Found IQ file: %s\n', filenameIQ);
    
    numFrames = 50;
    [dataIQ, header, tsIQ] = rdataread(fullPathIQ, numFrames);
    if (numFrames > header.frames)
        numFrames = header.frames;
    end
    
    num_lines = header.lines;
    width_mm = (num_lines - 1) * pitch_mm;
    x_axis = linspace(-width_mm/2, width_mm/2, num_lines);
    z_axis_iq = linspace(0, depth_mm, header.samples);
    
    idata = dataIQ(:,1:2:end,:);
    qdata = dataIQ(:,2:2:end,:);
    
    figure('Name', 'Clarius IQ Data');
    colormap(gray)
    
    for i = 1:numFrames
        I = squeeze(idata(i,:,:));
        Q = squeeze(qdata(i,:,:));
        BB = 20*log10(1 + sqrt(I.^2 + Q.^2));
        
        subplot(1,3,1);
        imagesc(x_axis, z_axis_iq, I, [-100 100]);
        title('I');
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        
        subplot(1,3,2);
        imagesc(x_axis, z_axis_iq, Q, [-100 100]);
        title('Q');
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        
        subplot(1,3,3);
        imagesc(x_axis, z_axis_iq, BB);
        title('Sample B for IQ');
        xlabel('Lateral Position (mm)');
        ylabel('Depth (mm)');
        axis image;
        daspect([1 1 1]);
        xlim([-width_mm/2, width_mm/2]);
        ylim([0, depth_mm]);
        drawnow;
    end
else
    fprintf('No IQ file present in folder. Skipping IQ processing.\n');
end

% Reset default window style
set(0, 'DefaultFigureWindowStyle', 'normal');


% =========================================================================
%% Local Helper Functions
% =========================================================================
function [data, header, ts] = rdataread(filename, frames)
    fid = fopen(filename, 'r');
    if (fid == -1)
        error('Cannot open file: %s', filename);
    end
    
    % Read header info
    hinfo = fread(fid, 5, 'int32');
    header.id = hinfo(1);
    header.frames = hinfo(2);
    header.lines = hinfo(3);
    header.samples = hinfo(4);
    header.sampleSize = hinfo(5);
    
    if frames > header.frames
        frames = header.frames;
    end
    
    if (header.id == 3 || header.id == 0) % PW IQ or IQ
        ts = zeros(1, frames, 'uint64');
        data = zeros(frames, header.samples*2, header.lines);
        for f = 1:frames
            ts(f) = fread(fid, 1, '*uint64');
            data(f,:,:) = fread(fid, [header.samples * 2, header.lines], 'int16');
        end
    elseif (header.id == 1) % ENV
        ts = zeros(1, frames, 'uint64');
        data = zeros(frames, header.samples, header.lines);
        for f = 1:frames
            ts(f) = fread(fid, 1, '*uint64');
            data(f,:,:) = fread(fid, [header.samples, header.lines], 'uint8');
        end
    elseif (header.id == 2) % RF
        ts = zeros(1, frames, 'uint64');
        data = zeros(frames, header.samples, header.lines);
        for f = 1:frames
            ts(f) = fread(fid, 1, '*uint64');
            data(f, :, :) = fread(fid, [header.samples, header.lines], 'int16');
        end
    else
        ts = [];
        data = [];
    end
    
    fclose(fid);
end
