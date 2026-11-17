// Build native Google Slides requests from the inspected A&M template.
// The Drive connector applies requests.json; image-jobs.json lists sidecars.
import fs from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const work = path.join(root, 'clarius_annotations/hoyt_presentation_v2_20261117');
const previous = JSON.parse(await fs.readFile(path.join(root, 'presentations/hoyt_content_v1.json'), 'utf8'));
for (const slide of previous.slides) {
  if (slide.image) slide.image = path.resolve(root, slide.image);
  if (slide.images) slide.images = slide.images.map(file => path.resolve(root, file));
}
const source = JSON.parse(await fs.readFile(path.join(work, 'raw-source.json'), 'utf8')).structuredContent;
const template = JSON.parse(await fs.readFile(path.join(work, 'raw-template.json'), 'utf8')).structuredContent;
const resultRoot = path.join(root, 'clarius_annotations/verified_evaluation_20261117');
const result = JSON.parse(await fs.readFile(path.join(resultRoot, 'scores/metrics.json'), 'utf8'));
const manifest = JSON.parse(await fs.readFile(path.join(resultRoot, 'reference_manifest.json'), 'utf8'));
const examples = JSON.parse(await fs.readFile(path.join(work, 'example-selection.json'), 'utf8'));
const old = structuredClone(previous.slides);
// Keep actual source text, notes and table cells, including user edits.
for (const s of old) {
  const live = source.slides.find(x => x.objectId === s.id);
  if (!live) throw new Error(`Source slide missing: ${s.id}`);
  const notesId = live.slideProperties.notesPage.notesProperties.speakerNotesObjectId;
  const notes = live.slideProperties.notesPage.pageElements.find(x => x.objectId === notesId);
  s.notes = notes?.shape?.text?.textElements?.map(x => x.textRun?.content || '').join('') || s.notes;
  const nativeTable = live.pageElements.find(x => x.table);
  if (nativeTable) {
    const cells = nativeTable.table.tableRows.map(r => r.tableCells.map(c => c.text.textElements.map(t => t.textRun?.content || '').join('').trim()));
    s.headers = cells[0]; s.rows = cells.slice(1);
  }
}
const labels = {medsam:'MedSAM', brightness_auto:'Brightness + auto', glcm:'GLCM + auto', combined_glcm:'Brightness + GLCM', glszm:'GLSZM + auto', brightness_glszm:'Brightness + GLSZM'};
const methods = Object.keys(labels);
const fmt = value => value.toFixed(3);
const auditNotes = 'Source: clarius_annotations/verified_evaluation_20261117/scores/metrics.json, reference_manifest.json and run_configuration.json. Evaluation completed 17 November 2026. References: PhD-student-verified contours according to the user. 98 images from 33 recorded examinations, excluded 006 and 045. Full-image prompt [0,0,1418,853], 8 gray levels, q0.985, GLSZM step (8,2). No settings tuned to these reference scores. Multiple captures are correlated. No patient-level independence or clinical performance claim.';
const summaryRows = methods.map(method => {
  const s = result.summary[method];
  return [labels[method], fmt(s.ssim.mean), fmt(s.dice.mean), fmt(s.iou.mean), String(result.rows.filter(r => r.method === method && r.dice === 0).length)];
});
const slides = [];
const add = s => slides.push(s);
const copy = (index, overrides={}) => add({...old[index], ...overrides, sourceSlideId:old[index].id});
copy(0,{title:'Clarius ultrasound\nliver segmentation',subtitle:'Sri Abhinav Thatavarthi\nResearch update for Dr. Hoyt\nNovember 2026',notes:'Planned presentation: 15–20 minutes, followed by discussion. Main slides cover background, our pipeline and the completed verified-reference comparison. Appendix retains the earlier development evidence. '+auditNotes});
const target = manifest.cases.find(c => c.id.startsWith('004_'));
copy(1,{image:path.join(resultRoot,'conversion_review',target.id+'.png'),caption:'PhD-student-verified contour, image 004. Visible lumen excluded.',notes:'The target is visible liver parenchyma, excluding surrounding tissue and clearly visible vessel lumens. This is a converted supplied verified reference, not one of the earlier AI draft outlines. '+auditNotes});
copy(2);
copy(3);
copy(4,{title:'Data and verified references',lines:['157 images from 57 recording folders.','98 verified references scored across 33 examinations.','006 and 045 excluded: no outline.'],caption:'Clean MATLAB inputs paired with supplied contours',notes:'The larger input set excludes folders named exam##. The user supplied 100 marked or unmarked images, of which 98 have contours. The PI is also revising segmentations. Distinguish that continuing effort from this completed PhD-student-verified subset. '+auditNotes});
copy(5);
copy(6);
copy(7,{steps:[['Spatial prompt','Interactive mode uses a drawn box. This evaluation uses the full image.'],['MedSAM proposal','One saved parent mask per image supplies every refinement.'],['Local appearance','Brightness statistics with GLCM or local GLSZM.'],['Automatic selection','A seed from the appearance groups guides tissue recovery.']],notes:old[7].notes+'\nThe evaluated protocol is a constant full-image box, approximating image-only use. MedSAM still requires that technical prompt. The reference supplies no box or seed. '+auditNotes});
copy(8,{lines:['Quantize into 8 gray levels.','Count pairs within masked windows.','Contrast, homogeneity, energy and correlation.','Four directions at two scales.']});
copy(9);
copy(10);
add({title:'Verified-reference test protocol',layout:'body',lines:['Same clean MATLAB env_gamma08 image for all methods.','Same full-image prompt and MedSAM parent.','Fixed defaults: 8 gray levels and q0.985.','Each prediction scored against its verified reference.'],foot:'588 image–method comparisons. No anatomical corrections or tuning during evaluation.',notes:auditNotes+' Contours are rasterized at the native 1418 by 853 grid. Technical gaps of up to 10 pixels are closed and logged. The user explicitly authorized the 75-pixel straight connector on image 021. Original marked files remain unchanged.'});
add({title:'Results on 98 verified images',layout:'table',headers:['Method','SSIM','Dice','IoU','Zero overlap'],rows:summaryRows,widths:[290,130,130,130,184],foot:'Per-image means. Zero overlap counts images with no shared foreground pixels.',notes:auditNotes+' Combined brightness + GLCM has the highest mean SSIM. Pure GLSZM has the highest mean Dice and IoU among these methods. Metric rankings differ. None establishes reliable automatic liver segmentation.'});
add({title:'GLCM and GLSZM combined with brightness',layout:'table',headers:['Method','SSIM','ROI SSIM','Dice','IoU'],rows:['combined_glcm','brightness_glszm'].map(m=>[labels[m],...['ssim','roi_mean_ssim','dice','iou'].map(k=>fmt(result.summary[m][k].mean))]),widths:[290,130,144,130,170],foot:'Brightness + GLCM is slightly higher on these aggregate metrics. Both have frequent localization errors.',notes:auditNotes+' Paired differences are descriptive. Full-frame SSIM includes shared background. ROI SSIM is the custom mean of the local SSIM map on the same union domain for every method. No significance test or independent-patient uncertainty estimate has been made.'});
for (const example of examples) add({title:example.role === 'upper_quartile'?'Example with substantial reference overlap':'Failure from selecting the wrong region',layout:'twoImages',labels:['Brightness + GLCM','Brightness + GLSZM'],images:example.images,foot:`Image ${example.id.slice(0,3)}. Green: verified contour. Orange: prediction. Dice ${fmt(example.scores.combined_glcm.dice)} and ${fmt(example.scores.brightness_glszm.dice)}.`,notes:auditNotes+' Example: '+example.id+'. Selection rule: '+(example.role==='upper_quartile'?'75th percentile of Combined GLCM Dice, deterministic case tie-break.':'First Combined GLCM zero-overlap image in evaluation order.')+' These are illustrative cases, not substitutes for the full set.'});
add({title:'Limits of the full-image workflow',layout:'body',lines:['MedSAM often covers almost the entire field.','Texture similarity can select a different anatomical region.','Combined GLCM has zero overlap in 38 of 98 images.','Shared background can raise SSIM despite poor liver overlap.'],foot:'No algorithm exceptions occurred. Anatomical selection remains the main failure.',notes:auditNotes+' MedSAM mean precision is '+fmt(result.summary.medsam.precision.mean)+' and recall '+fmt(result.summary.medsam.recall.mean)+'. This supports broad coverage with excess tissue. Pure GLSZM has 12 zero-overlap images, brightness + GLSZM 36, and pure GLCM 43. These are valid but inaccurate predictions, not runtime exceptions.'});
add({title:'Next evaluation and implementation steps',layout:'body',lines:['Review disagreements with the PI and PhD student.','Add reliable liver localization before texture refinement.','Test fixed full-image and realistic interactive prompts separately.','Reserve patients or acquisitions for evaluation before tuning.'],foot:'PI revisions and additional verified image–mask pairs will support the next round.',notes:auditNotes+' Use distinct development and evaluation groups when identifiers allow. Test a detector or ultrasound-adapted segmentation model for localization. A reference-derived tight box would measure assisted segmentation using reference information and must be labeled separately. Current results have now been inspected and should not be treated as an untouched final test set after further tuning.'});
copy(19,{lines:['Establish reliable liver regions first.','Study variation within the liver.','Collect expert lesion labels before testing detection.'],notes:old[19].notes+'\nReliable liver localization remains necessary based on the current 98-image evaluation. Lesion localization requires separate expert lesion labels. Task-specific training and ultrasound-adapted models remain future comparisons.'});
copy(20,{title:'Discussion with Dr. Hoyt',lines:['Confirm the liver and vessel-lumen target.','Prioritize localization before further texture tuning.','Agree on a primary overlap metric and evaluation grouping.','Plan the next comparison with PI-reviewed references.'],notes:old[20].notes+'\n'+auditNotes});
add({title:'Questions and discussion',layout:'closing',notes:'End of the 15–20 minute main presentation. Appendix slides retain the earlier development experiments, settings and metric definitions. The current 98-image verified comparison is the main evidence.'});
for (const i of [11,12,13,14,15]) copy(i,{title:'Appendix: '+old[i].title,foot:(old[i].foot||'')+' Historical development data.',notes:old[i].notes+'\nHistorical exploratory results. Different images, provisional references and targeted prompts prevent direct comparison with the new 98-image full-image evaluation.'});
add({title:'Appendix: reference conversion audit',layout:'body',lines:['Retain the largest enclosed liver contour and internal exclusions.','Close and log technical seams up to 10 pixels.','Connect image 021 endpoints with the authorized straight line.','Verify native dimensions and unchanged source hashes.'],foot:'100 original annotated files preserved. 98 reference masks scored.',notes:auditNotes+' The conversion method assigns stroke pixels to the nearest contour compartment. It neither invents anatomy nor uses prior AI masks. Binary reference PNGs are expected to appear black and white. The review viewer overlays their boundaries on the ultrasound image.'});
copy(21,{title:'Appendix: current parameter settings',rows:old[21].rows.map((r,i)=>i===0?[r[0],'MedSAM ViT-B. This test uses the fixed full-image box.']:r)});
copy(22,{title:'Appendix: interpretation of the metrics'});
copy(23,{title:'Appendix: historical per-frame scores',notes:old[23].notes+'\nThese seven-frame provisional results are retained for historical context only. They are superseded for the automatic current-input protocol by the 98-image verified-reference evaluation.'});
copy(24,{title:'Appendix: selected references'});

const requests = [], images = [];
const color = hex => ({red:parseInt(hex.slice(1,3),16)/255,green:parseInt(hex.slice(3,5),16)/255,blue:parseInt(hex.slice(5,7),16)/255});
const maroon=color('#500000'), ink=color('#252525'), gray=color('#626262');
const geom = (sid,x,y,w,h) => ({pageObjectId:sid,size:{width:{magnitude:w,unit:'PT'},height:{magnitude:h,unit:'PT'}},transform:{scaleX:1,scaleY:1,translateX:x,translateY:y,unit:'PT'}});
function move(id,x,y,w,h) { requests.push({updatePageElementTransform:{objectId:id,applyMode:'ABSOLUTE',transform:{scaleX:w/(3000000/12700),scaleY:h/(3000000/12700),translateX:x,translateY:y,unit:'PT'}}}); }
function replace(id,value) {
  // The content exemplar's placeholders are empty. Slides rejects deleteText
  // on an empty shape, whereas the cover and ending contain sample text.
  if (!id.includes('_p2_')) requests.push({deleteText:{objectId:id,textRange:{type:'ALL'}}});
  requests.push({insertText:{objectId:id,text:value}});
}
function style(id,size,fill=ink,bold=false,font='Open Sans') { requests.push({updateTextStyle:{objectId:id,textRange:{type:'ALL'},style:{fontFamily:font,fontSize:{magnitude:size,unit:'PT'},foregroundColor:{opaqueColor:{rgbColor:fill}},bold},fields:'fontFamily,fontSize,foregroundColor,bold'}}); }
function text(sid,id,value,x,y,w,h,size=22,fill=ink,bold=false) {
  requests.push({createShape:{objectId:id,shapeType:'TEXT_BOX',elementProperties:geom(sid,x,y,w,h)}},{insertText:{objectId:id,text:value}}); style(id,size,fill,bold);
  requests.push({updateParagraphStyle:{objectId:id,textRange:{type:'ALL'},style:{spaceAbove:{magnitude:0,unit:'PT'},spaceBelow:{magnitude:5,unit:'PT'},lineSpacing:108},fields:'spaceAbove,spaceBelow,lineSpacing'}});
}
function table(sid,id,headers,rows,widths,x=48,y=185,w=864,h=260,font=18) {
  const cells=[headers,...rows];
  requests.push({createTable:{objectId:id,rows:cells.length,columns:headers.length,elementProperties:geom(sid,x,y,w,h)}});
  const total=widths.reduce((a,b)=>a+b,0);
  widths.forEach((v,c)=>requests.push({updateTableColumnProperties:{objectId:id,columnIndices:[c],tableColumnProperties:{columnWidth:{magnitude:v/total*w,unit:'PT'}},fields:'columnWidth'}}));
  for(let r=0;r<cells.length;r++) for(let c=0;c<cells[r].length;c++) {
    requests.push({insertText:{objectId:id,cellLocation:{rowIndex:r,columnIndex:c},text:String(cells[r][c])}},
      {updateTextStyle:{objectId:id,cellLocation:{rowIndex:r,columnIndex:c},textRange:{type:'ALL'},style:{fontFamily:'Open Sans',fontSize:{magnitude:font,unit:'PT'},bold:r===0,foregroundColor:{opaqueColor:{rgbColor:r===0?color('#ffffff'):ink}}},fields:'fontFamily,fontSize,bold,foregroundColor'}},
      {updateParagraphStyle:{objectId:id,cellLocation:{rowIndex:r,columnIndex:c},textRange:{type:'ALL'},style:{spaceAbove:{magnitude:0,unit:'PT'},spaceBelow:{magnitude:0,unit:'PT'},lineSpacing:100},fields:'spaceAbove,spaceBelow,lineSpacing'}},
      {updateTableCellProperties:{objectId:id,tableRange:{location:{rowIndex:r,columnIndex:c},rowSpan:1,columnSpan:1},tableCellProperties:{tableCellBackgroundFill:{solidFill:{color:{rgbColor:r===0?maroon:color(r%2?'#ffffff':'#f2eeee')},alpha:1}},contentAlignment:'MIDDLE'},fields:'tableCellBackgroundFill,contentAlignment'}});
  }
}
async function image(sid,id,file,x,y,w,h) {
  // PNG dimensions from IHDR. All scientific figures use full fit, no crop.
  const data=await fs.readFile(file);
  let width,height;
  if(data.subarray(1,4).toString()==='PNG') {width=data.readUInt32BE(16);height=data.readUInt32BE(20);}
  else throw new Error('Expected a PNG scientific image: '+file);
  const scale=Math.min(w/width,h/height);
  images.push({file,request:{createImage:{objectId:id,url:file,elementProperties:geom(sid,x+(w-width*scale)/2,y+(h-height*scale)/2,width*scale,height*scale)}}});
}
for(let i=0;i<slides.length;i++) {
  const s=slides[i],sid=`hoytv2_s${String(i+1).padStart(2,'0')}`;
  s.id=sid; s.narrativeRole=i===0?'cover':s.layout;
  const exemplar=i===0?'p1':s.layout==='closing'?'p4':'p2';
  s.templateExemplar=exemplar; s.construction='duplicate inspected exemplar';
  s.mediaMapping=exemplar==='p4'?'Keep A&M photograph and logo inherited from layout':'Keep A&M banner, logo and background inherited from layout. No slide-local exemplar media.';
  const templateSlide=template.slides.find(x=>x.objectId===exemplar);
  const mapping={[exemplar]:sid};
  for(const e of templateSlide.pageElements) mapping[e.objectId]=sid+'_'+e.objectId;
  requests.push({duplicateObject:{objectId:exemplar,objectIds:mapping}});
  if(i===0) {
    replace(mapping.p1_i2,s.title); move(mapping.p1_i2,72,218,816,118);
    replace(mapping.p1_i3,s.subtitle); move(mapping.p1_i3,144,340,672,112); style(mapping.p1_i3,22,color('#ffffff'),true);
    continue;
  }
  if(s.layout==='closing') { replace(mapping.p4_i4,s.title); continue; }
  const title=mapping.p2_i3,left=mapping.p2_i4,right=mapping.p2_i5;
  replace(title,s.title);
  // Long appendix titles need a local reduction from the inherited 44 pt.
  if(s.title.length>49) style(title,35,maroon,false,'Oswald');
  const body=id=>requests.push({deleteObject:{objectId:id}});
  if(s.layout==='body'||s.layout==='references') {
    body(right);move(left,48,185,864,285);replace(left,s.lines.join('\n'));
    style(left,s.layout==='references'?20:22,maroon,true);
    requests.push({createParagraphBullets:{objectId:left,textRange:{type:'ALL'},bulletPreset:'BULLET_DISC_CIRCLE_SQUARE'}},
      {updateParagraphStyle:{objectId:left,textRange:{type:'ALL'},style:{spaceBelow:{magnitude:16,unit:'PT'},lineSpacing:110},fields:'spaceBelow,lineSpacing'}});
  } else {
    body(left);body(right);
    if(s.layout==='imageRight') {
      text(sid,sid+'_body',s.lines.join('\n\n'),48,185,335,285,22,maroon,true);
      await image(sid,sid+'_img',s.image,415,185,497,263);
      text(sid,sid+'_caption',s.caption,415,460,497,48,14,gray);
    } else if(s.layout==='twoImages'||s.layout==='threeImages') {
      const n=s.images.length,gap=24,w=(864-gap*(n-1))/n;
      for(let j=0;j<n;j++) {
        text(sid,sid+'_label'+j,s.labels[j],48+j*(w+gap),181,w,48,21,maroon,true);
        await image(sid,sid+'_img'+j,s.images[j],48+j*(w+gap),235,w,n===2?238:163);
        if(s.descriptions) text(sid,sid+'_description'+j,s.descriptions[j],48+j*(w+gap),400,w,70,18);
      }
    } else if(s.layout==='process') {
      s.steps.forEach(([name,description],j)=>{
        text(sid,sid+'_name'+j,name,48,185+j*71,240,65,22,maroon,true);
        text(sid,sid+'_desc'+j,description,305,185+j*71,607,65,22);
      });
    } else if(s.layout==='glcm') {
      text(sid,sid+'_body',s.lines.join('\n\n'),48,181,444,292,22,maroon,true);
      table(sid,sid+'_matrix',s.headers,s.rows,[1,1,1],530,235,382,170,18);
    } else if(s.layout==='table') table(sid,sid+'_table',s.headers,s.rows,s.widths,48,185,864,Math.min(285,(s.rows.length+1)*40),s.rows.length>=7?16:18);
    else throw new Error('Unsupported layout '+s.layout);
  }
  if(s.foot) text(sid,sid+'_foot',s.foot,48,488,864,39,14,gray);
}
await fs.writeFile(path.join(work,'slide-plan.json'),JSON.stringify({slides,sources:previous.sources},null,2));
await fs.writeFile(path.join(work,'requests.json'),JSON.stringify(requests));
await fs.writeFile(path.join(work,'image-jobs.json'),JSON.stringify(images,null,2));
console.log(JSON.stringify({slides:slides.length,mainSlides:21,appendixSlides:slides.length-21,requests:requests.length,images:images.length}));
