function audit = test_clarius_glcm_physical(outputPath)
% Independent pair enumeration + MATLAB graycomatrix + known physical grids.
arguments
    outputPath (1,:) char = ''
end
tests=0; maxError=0; offsets=[0 1;-1 0;-1 1;-1 -1;0 -2];
[f,d,c]=clarius_glcm_direction(ones(9),8,[0 1]);
assert(isequal(f,[0 0]) && d.valid_pairs==72 && full(sum(c(:)))==144);tests=tests+1;
q=1+mod(sum(indices(9,9),3),2);
f=clarius_glcm_direction(q,2,[0 1]);assert(max(abs(f-[1 1]))<1e-12);tests=tests+1;
f=clarius_glcm_direction(q,2,[-1 1]);assert(max(abs(f-[0 1]))<1e-12);tests=tests+1;
rng(7); q=randi(4,9,11);
for d=1:size(offsets,1)
    [f,~,counts]=clarius_glcm_direction(q,4,offsets(d,:));
    expected=graycomatrix(q,'NumLevels',4,'GrayLimits',[1 4],'Offset',offsets(d,:),'Symmetric',true);
    assert(isequal(full(counts),expected),'Counts differ from MATLAB graycomatrix');
    properties=graycoprops(expected,'Contrast');
    p=expected(:)/sum(expected(:)); p=p(p>0);
    truth=[properties.Contrast -sum(p.*log2(p))];
    error=max(abs(f-truth));maxError=max(maxError,error);assert(error<1e-12);tests=tests+1;
end
q(3:5,4:7)=NaN;q(1,1)=NaN;
for d=1:size(offsets,1)
    [f,~,counts]=clarius_glcm_direction(q,4,offsets(d,:));
    expected=zeros(4);
    for r=1:size(q,1)
        for col=1:size(q,2)
            rr=r+offsets(d,1);cc=col+offsets(d,2);
            if rr<1||rr>size(q,1)||cc<1||cc>size(q,2),continue;end
            a=q(r,col);b=q(rr,cc);
            if isfinite(a)&&isfinite(b),expected(a,b)=expected(a,b)+1;end
        end
    end
    expected=expected+expected.';
    assert(isequal(full(counts),expected),'Masked pair counts differ from enumeration');
    [i,j]=ndgrid(1:4);p=expected/sum(expected(:));
    truth=[sum(p(:).*(i(:)-j(:)).^2), -sum(p(p>0).*log2(p(p>0)))];
    error=max(abs(f-truth));maxError=max(maxError,error);assert(error<1e-12);tests=tests+1;
end
[f,d]=clarius_glcm_direction(nan(4),8,[0 1]);assert(all(isnan(f))&&d.valid_pairs==0);tests=tests+1;
for levels=[8 16 32 64]
    q=1+(levels-1)*mod(sum(indices(9,9),3),2);
    f=clarius_glcm_direction(q,levels,[0 1]);
    assert(abs(f(1)/(levels-1)^2-1)<1e-12 && abs(f(2)-1)<1e-12);tests=tests+1;
end
z=(0:999)'*.013; x=(0:24)*.13;
[constant,zIso,xIso,spacing]=clarius_isotropic_envelope(ones(1000,25)*2,z,x,.13);
assert(max(abs(constant(:)-2))<1e-10 && max(abs(diff(zIso)-.13))<1e-10 && max(abs(diff(xIso)-.13))<1e-10);
assert(spacing.antialias_filter_order(1)>0 && spacing.antialias_filter_order(2)==0);tests=tests+1;
input=2+sin(2*pi*20*z)+zeros(1,25);
filtered=clarius_isotropic_envelope(input,z,x,.13);[Z,X]=ndgrid(zIso,xIso);
unfiltered=interp2(x,z,input,X,Z,'linear');
filteredStd=std(filtered(10:end-10,:),0,'all'); unfilteredStd=std(unfiltered(10:end-10,:),0,'all');
assert(filteredStd<unfilteredStd*.1,'Anti-alias filtering failed to reject out-of-band modulation');tests=tests+1;
failed=false;try,clarius_isotropic_envelope(input,z,x,.065);catch,failed=true;end
assert(failed,'Lateral upsampling must be rejected');tests=tests+1;
env=2+(.1*sin((0:170)'/7))+(.03*(0:24));
mask=true(size(env));mask(:,1:5)=false;
r=clarius_glcm_physical(env,(0:170)'*.013,x,'InputDomain','Envelope', ...
    'TransmitFrequencyHz',16e6,'Levels',[8 64],'WindowPulseLengths',[2 4],'RoiMask',mask);
assert(numel(r.configurations)==8 && numel(r.feature_names)==2 && strcmp(r.direction_aggregation,'NONE'));tests=tests+1;
assert(abs(r.wavelength_mm-.09625)<1e-12 && abs(r.pulse_length_mm-.1925)<1e-12);tests=tests+1;
assert(isequaln(r.configurations{1}.quantized_image,r.configurations{3}.quantized_image),'Windows must share image-wide quantization');tests=tests+1;
assert(all(isnan(r.configurations{1}.quantized_image(:,1:5)),'all'),'ROI background must be excluded');tests=tests+1;
env=repmat([0 1e-6 1e-3 10^(-29/20) 1 1 1 1 1],9,1);
r=clarius_glcm_physical(env,(0:8)'*.13,(0:8)*.13,'InputDomain','Envelope', ...
    'TransmitFrequencyHz',16e6,'Levels',8,'WindowPulseLengths',2);
q=r.configurations{2}.quantized_image;
assert(isnan(q(1,1)) && isequal(q(1,2:5),[1 1 5 8]),'B-mode fixed-range bins or invalid-zero convention differ');tests=tests+1;
assert(abs(r.bmode_dB(1,4)+29)<1e-12 && r.bmode_dB(1,2)==-60,'B-mode must use amplitude 20log10 and clipping');tests=tests+1;
assert(r.configurations{1}.quantized_image(1,3)==1,'Linear envelope must not receive logarithmic quantization');tests=tests+1;
temporary=tempname;mkdir(temporary);cleanup=onCleanup(@() rmdir(temporary,'s'));
raw=fullfile(temporary,'fixture_rf.raw');yaml=fullfile(temporary,'fixture_rf.yml');
fid=fopen(raw,'w','ieee-le');fwrite(fid,uint32([4 1 4 64 2]),'uint32');
stamp=bitshift(uint64(1),53)+uint64(17);fwrite(fid,stamp,'uint64');
samples=int16(reshape(1:256,64,4));fwrite(fid,samples,'int16');fclose(fid);
text=sprintf('probe:\n  pitch: 130.00\nsampling rate: 60 MHz\ntransmit frequency: 16 MHz\nsize: {samples per line: 64,number of lines: 4,sample size: 2 bytes}\ndelay samples: 24\nlines:\n');
for k=0:3,text=[text sprintf('  - {rx element: %d,tx element: %d,angle: 0 degrees}\n',k,k)];end
fid=fopen(yaml,'w');fprintf(fid,'%s',text);fclose(fid);
[read,z,x,m]=read_clarius_rf_physical(raw,yaml);
assert(isequal(read,double(samples)) && strcmp(m.timestamp_raw,sprintf('%u',stamp)));tests=tests+1;
assert(abs(diff(z(1:2))-1540/(2*60e6)*1000)<1e-12 && abs(diff(x(1:2))-.13)<1e-12);tests=tests+1;
text=strrep(text,'angle: 0','angle: 5');fid=fopen(yaml,'w');fprintf(fid,'%s',text);fclose(fid);
failed=false;try,read_clarius_rf_physical(raw,yaml);catch,failed=true;end
assert(failed,'Steered geometry must not silently use Cartesian axes');tests=tests+1;
audit=struct('tests_passed',tests,'max_feature_reference_error',maxError, ...
    'filtered_out_of_band_std',filteredStd,'unfiltered_out_of_band_std',unfilteredStd, ...
    'features',{{'Contrast','Entropy'}},'matlab_version',version);
if ~isempty(outputPath)
    fid=fopen(outputPath,'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(audit));fclose(fid);
end
fprintf('Physical GLCM: %d checks passed, maximum reference error %.3g\n',tests,maxError);
end

function result=indices(h,w)
[r,c]=ndgrid(0:h-1,0:w-1);result=cat(3,r,c);
end
