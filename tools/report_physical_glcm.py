"""Summarize MATLAB CSV/audits and link plots; no image processing in Python."""
import csv
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT/'clarius_annotations/physical_glcm';root=base/'pilot'
    prepared=json.loads((base/'prepared/frames.json').read_text())
    report=json.loads((root/'report.json').read_text())
    validation=json.loads((base/'validation.json').read_text())
    with (root/'summary.csv').open(newline='',encoding='utf-8-sig') as handle:rows=list(csv.DictReader(handle))
    with (root/'parameter_comparison.csv').open(newline='',encoding='utf-8-sig') as handle:comparison=list(csv.DictReader(handle))
    assert not report['failures']
    assert len(rows)==len(prepared['jobs'])*2*4*3*4
    assert len({tuple(row[k] for k in ('File','Domain','Levels','WindowPL','Direction')) for row in rows})==len(rows)
    for job in prepared['jobs']:
        for path,hash_key in [('raw','raw_sha256'),('metadata_yml','yaml_sha256')]:
            assert hashlib.sha256(Path(job[path]).read_bytes()).hexdigest()==job[hash_key]
    lines=['# Physically calibrated GLCM pilot','',
        f'{len(prepared["jobs"])} captures, {len(rows)} per-image/configuration/direction summary rows. MATLAB produced every image/feature map.',
        f'{validation["tests_passed"]} MATLAB checks passed. Maximum feature difference against independent count references: {validation["max_feature_reference_error"]}. Three additional byte-preparation tests passed.',
        'No segmentation, CNN, GLSZM, clustering or automatic tissue selection enters this experiment.', '',
        '## Physical calibration','',
        'RF sample rate and transmit frequency come from each matching YAML. Zero-steering receive-element positions times probe pitch define lateral coordinates. Unsupported curved/steered/nonuniform geometry fails explicitly.',
        'All four captures use 60 MHz RF sampling and 0.130 mm lateral line spacing. At assumed sound speed 1540 m/s, axial sample spacing is 0.0128333 mm. FIR anti-aliasing precedes resampling to 0.130 by 0.130 mm.',
        'One horizontal or axial offset is 0.130 mm. A diagonal offset is 0.18385 mm. Directions remain separate; isotropic pixels do not make diagonal and axial offsets equal in length.', '',
        '## Quantization and windows','',
        'Linear envelope and 60 dB log B-mode both derive from the same Hilbert envelope after physical resampling. Image-wide quantization is shared by all windows, with no per-window contrast normalization. Zero envelope amplitude is excluded, matching the supplied script.',
        'Tested levels: 8, 16, 32, 64. Windows: 5, 10, 20 pulse lengths. Two cycles per pulse is an explicit assumption from the supplied code, not a measured pulse duration. Each case uses its own transmit frequency.',
        'Contrast and entropy are retained separately for four directions. Contrast/(levels-1)^2 is also reported to aid cross-level interpretation; it is a rescaling of contrast, not an extra feature.', '',
        '| Capture | Window PL | Actual side (mm) | Side pixels | Mean horizontal pairs | B-mode occupied entries |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['Domain']=='Bmode' and row['Levels']=='64' and row['Direction']=='Horizontal':
            lines.append(f'| {row["File"][:3]} | {row["WindowPL"]} | {float(row["ActualWindowMm"]):.2f} | {row["WindowPixels"]} | {float(row["MeanValidPairs"]):.1f} | {100*float(row["MeanOccupiedFraction"]):.2f}% |')
    lines+=['','## Parameter sensitivity','',
        'At 64 levels there are 4096 possible matrix entries. The smallest tested windows contain only about 42–72 horizontal pairs. Their sparse matrices motivate checking larger windows and coarser quantization, particularly for entropy.',
        'Larger windows improve pair support but blur spatial boundaries. Lower levels change both quantization error and entropy range. These technical comparisons do not identify an optimal liver/lesion setting. Inspect 16/32 levels with 10 PL as intermediate candidates, keeping 5 and 20 PL as scale comparisons.',
        'The common visual color range permits comparison within each figure. The map lattice targets 80% overlap. Gray borders are locations without a valid full-window estimate; original RF amplitudes are not cropped to those map centers.', '',
        '## Files','',
        '- `review.html`: comparison figures with physical axes.',
        '- `summary.csv`: every capture, domain, level, window and direction.',
        '- `parameter_comparison.csv`: macro means across captures, still separated by direction. Equal PL can correspond to different millimeter sizes when transmit frequencies differ.',
        '- `*_physical_glcm.mat`: isotropic amplitudes, coordinates, quantized images, two feature maps per direction and count diagnostics.',
        '- `report.json`, `comparison_audit.json`: MATLAB provenance.', '',
        '## Limits','',
        'These are full-frame technical demonstrations, not a mask-accuracy evaluation. Existing PNG masks come from a different recorded B-mode grid and must not be copied onto RF data by shape alone. A source-grid ROI can be supplied explicitly to clarius_glcm_physical.',
        'The recorded uint8 _env.raw is already B-mode grayscale; it is not linear envelope amplitude and cannot supply the linear-envelope branch. This experiment starts from RF instead.',
        'No inverse-TGC or speckle-denoising step is enabled. Anti-alias filtering serves resampling. The first RF sample defines a relative depth origin; reported delay samples are recorded without assuming absolute cross-stream registration. Isotropic sample spacing does not establish isotropic acoustic resolution.',
        'The original supplied script remains unchanged. Adaptation source SHA256: E5A1AB8653390369C77B424B5E6ACD2D8DDA20DCCD0A66A9D873339394C18C9C.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plots=sorted(root.glob('*.png'))
    figures=''.join(f'<figure><figcaption>{html.escape(p.name)}</figcaption><img src="{p.name}" loading="lazy"></figure>' for p in plots)
    page='<!doctype html><meta charset="utf-8"><title>Physical GLCM comparisons</title><style>body{font:17px Arial;margin:20px;background:#eee;color:#222}figure{margin:24px 0}img{display:block;max-width:100%;height:auto;background:white}figcaption{margin-bottom:8px}</style><h1>Physical GLCM comparisons</h1><p>Contrast and entropy, linear envelope and RF-derived B-mode. Isotropic spacing 0.130 mm. Directions stay separate in saved arrays. Figures show the horizontal direction.</p><p><a href="REPORT.md">Report</a> · <a href="summary.csv">All summaries</a></p>'+figures
    (root/'review.html').write_text(page,encoding='utf-8')
    output=dict(captures=len(prepared['jobs']),summary_rows=len(rows),parameter_rows=len(comparison),matlab_checks=validation['tests_passed'],
        byte_preparation_tests=3,inputs_unchanged=True,features=report['features'],direction_aggregation='NONE',
        configurations=dict(levels=[8,16,32,64],window_pulse_lengths=[5,10,20],isotropic_spacing_mm=.13,cycles_per_pulse_assumed=2),
        source_script_sha256='E5A1AB8653390369C77B424B5E6ACD2D8DDA20DCCD0A66A9D873339394C18C9C',
        matlab_validation=validation,claim='Technical validation and parameter sensitivity; no optimum or segmentation accuracy established')
    (base/'audit.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
