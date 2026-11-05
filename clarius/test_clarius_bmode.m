function test_clarius_bmode
% Technical checks only; not an anatomical/image-quality validation.
native = uint8([0 16 64 128 255]);
[displayed,toned] = render_clarius_bmode(native,[1 5]);
assert(isequal(toned,uint8([0 28 84 147 255])),'Known gamma LUT entries differ');
assert(isequal(displayed,toned),'Native-size export changed samples');
[~,identity] = render_clarius_bmode(native,[1 5],1);
assert(isequal(identity,native),'Gamma 1 must preserve uint8 values');
zero = render_clarius_bmode(zeros(24,8,'uint8'),[853 1418]);
assert(isequal(size(zero),[853 1418]) && ~any(zero(:)),'Black/grid preservation failed');
constant = render_clarius_bmode(repmat(uint8(64),17,9),[31 45]);
assert(all(constant(:)==84),'Resize changed a constant tone');

temporary = tempname;mkdir(temporary);
cleanup = onCleanup(@() rmdir(temporary,'s'));
raw = fullfile(temporary,'fixture.raw');
fid = fopen(raw,'w','ieee-le');
fwrite(fid,uint32([1 1 5 2 1]),'uint32');
timestamp = bitshift(uint64(1),53)+uint64(123);
fwrite(fid,timestamp,'uint64');
env = uint8([0 16 64 128 255;255 128 64 16 0]);
fwrite(fid,env,'uint8');fclose(fid);
job = struct('id','001_fixture','raw',raw,'raw_sha256','synthetic', ...
    'source_archive','synthetic','source_member','fixture_env.raw', ...
    'selected_frame',1,'timestamp_raw',sprintf('%u',timestamp),'timestamp_delta_ns','0');
manifest = struct('jobs',job,'failures',[],'expected_captures',1);
manifestPath = fullfile(temporary,'frames.json');
fid=fopen(manifestPath,'w');fprintf(fid,'%s',jsonencode(manifest));fclose(fid);
result = export_clarius_bmode(manifestPath,fullfile(temporary,'exported'));
actual = imread(fullfile(temporary,'exported','images','001_fixture.png'));
assert(isequal(actual,render_clarius_bmode(env)),'RAW orientation or export differs');
assert(strcmp(result.cases{1}.timestamp_raw,sprintf('%u',timestamp)), ...
    'Lost integer timestamp precision');
fprintf('MATLAB B-mode checks passed: gamma LUT, identity, black, dimensions, constant, RAW orientation and timestamp\n');
end
