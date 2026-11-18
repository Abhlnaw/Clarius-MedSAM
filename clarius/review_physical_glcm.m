function audit = review_physical_glcm(outputPath)
% Compare feature sampling/configuration without assigning tissue identities.
arguments
    outputPath (1,:) char
end
T=readtable(fullfile(outputPath,'summary.csv'),'TextType','string');
keys=unique(T(:,{'Domain','Levels','WindowPL','Direction'}),'rows');
rows=cell(height(keys),1);
for k=1:height(keys)
    match=T.Domain==keys.Domain(k)&T.Levels==keys.Levels(k)&T.WindowPL==keys.WindowPL(k)&T.Direction==keys.Direction(k);
    row=table2struct(keys(k,:));row.Images=nnz(match);
    fields={'ActualWindowMm','WindowPixels','MeanValidPairs','MeanOccupiedFraction','MeanNormalizedContrast','MeanEntropyBits'};
    for f=1:numel(fields),row.(fields{f})=mean(T.(fields{f})(match),'omitnan');end
    rows{k}=row;
end
comparison=struct2table(vertcat(rows{:}));
writetable(comparison,fullfile(outputPath,'parameter_comparison.csv'));
files=dir(fullfile(outputPath,'*_physical_glcm.mat'));
for k=1:numel(files)
    loaded=load(fullfile(files(k).folder,files(k).name),'result');result=loaded.result;
    [rf,~,~,source]=read_clarius_rf_physical(result.source.raw,result.source.yaml);
    assert(isequal(size(rf),result.source.native_size) && source.sample_rate_Hz==result.source.sample_rate_Hz, ...
        'Saved result differs from currently validated RF metadata');
    for feature=1:2
        plot_grid(result,64,[5 10 20],feature,fullfile(outputPath,sprintf('%s_%s_windows.png',files(k).name(1:3),result.feature_names{feature})));
        if k==1
            plot_grid(result,[8 16 32 64],10,feature,fullfile(outputPath,sprintf('%s_%s_levels.png',files(k).name(1:3),result.feature_names{feature})));
        end
    end
end
audit=struct('images',numel(files),'settings',height(comparison),'features',{{'Contrast','Entropy'}}, ...
    'directions_kept_separate',true,'plot_direction','Horizontal only for readable comparisons; all four directions retained in MAT/CSV', ...
    'claim','Technical calibration and parameter-sensitivity pilot; no optimal setting or segmentation accuracy established');
fid=fopen(fullfile(outputPath,'comparison_audit.json'),'w');assert(fid~=-1);fprintf(fid,'%s',jsonencode(audit));fclose(fid);
end

function plot_grid(result,levels,windows,feature,destination)
domains={'Envelope','Bmode'};settings=max(numel(levels),numel(windows));
configs=cell(2,settings);maximum=0;
for domain=1:2
    for setting=1:settings
        level=levels(min(setting,numel(levels)));window=windows(min(setting,numel(windows)));
        found=find(cellfun(@(c) strcmp(c.domain,domains{domain})&&c.levels==level&&c.window_pulse_lengths==window,result.configurations),1);
        configs{domain,setting}=result.configurations{found};
        values=configs{domain,setting}.feature_maps(:,:,1,feature);
        if feature==1,values=values/(level-1)^2;end
        valid=values(isfinite(values));if ~isempty(valid),maximum=max(maximum,max(valid));end
    end
end
fig=figure('Visible','off','Color','w','Position',[0 0 400*settings 780]);
cleanup=onCleanup(@() close(fig));
layout=tiledlayout(2,settings,'TileSpacing','compact','Padding','compact');
for domain=1:2
    for setting=1:settings
        c=configs{domain,setting};values=c.feature_maps(:,:,1,feature);
        if feature==1,values=values/(c.levels-1)^2;end
        ax=nexttile;imagesc(ax,c.x_centers_mm,c.z_centers_mm,values,'AlphaData',isfinite(values));
        set(ax,'Color',[.85 .85 .85],'XColor',[.15 .15 .15],'YColor',[.15 .15 .15]);
        axis(ax,'image');clim(ax,[0 max(maximum,eps)]);cb=colorbar(ax);cb.Color=[.15 .15 .15];
        xlim(ax,[result.xIso_mm(1) result.xIso_mm(end)]);ylim(ax,[result.zIso_mm(1) result.zIso_mm(end)]);
        title(ax,sprintf('%s: L=%d, %g PL (%.2f mm)',domains{domain},c.levels,c.window_pulse_lengths,c.actual_window_mm),'Color','k');
        xlabel(ax,'Lateral (mm)','Color','k');ylabel(ax,'Axial (mm)','Color','k');
    end
end
label=result.feature_names{feature};if feature==1,label='Contrast / (levels-1)^2';else,label='Entropy (bits)';end
title(layout,sprintf('%s | Horizontal offset %.3f mm | Gray: no valid full-window estimate',label,result.offset_distances_mm(1)),'Color','k');
exportgraphics(fig,destination,'Resolution',120);
end
