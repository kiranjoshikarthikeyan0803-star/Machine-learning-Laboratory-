/**
 * OncoVision AI — Client Application Controller
 * Handles image pipeline, live SVG charts, training arena, and in-browser code compiler.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State Store
    const state = {
        currentImageSource: null,
        currentSampleId: null,
        uploadedFile: null,
        activeAlgorithm: 'random_forest',
        filterType: 'bilateral',
        claheClip: 2.0,
        codeTemplates: {},
        isProcessing: false,
        isTraining: false
    };

    // DOM Elements
    const elements = {
        // Nav & Header
        engineStatus: document.getElementById('engineStatus'),
        activeModelLabel: document.getElementById('activeModelLabel'),
        datasetCountLabel: document.getElementById('datasetCountLabel'),
        btnRunFullPipeline: document.getElementById('btnRunFullPipeline'),
        btnToggleSandbox: document.getElementById('btnToggleSandbox'),
        btnOpenDocsModal: document.getElementById('btnOpenDocsModal'),
        
        // Stepper
        stepItems: document.querySelectorAll('.step-item'),
        
        // Stage 1
        dropZone: document.getElementById('dropZone'),
        imageInput: document.getElementById('imageInput'),
        samplesContainer: document.getElementById('samplesContainer'),
        activeImageMetaBar: document.getElementById('activeImageMetaBar'),
        metaFileName: document.getElementById('metaFileName'),
        metaDimensions: document.getElementById('metaDimensions'),
        
        // Stage 2
        filterTypeSelect: document.getElementById('filterTypeSelect'),
        claheClipSlider: document.getElementById('claheClipSlider'),
        claheVal: document.getElementById('claheVal'),
        btnReProcess: document.getElementById('btnReProcess'),
        previewOriginal: document.getElementById('previewOriginal'),
        previewGrayscale: document.getElementById('previewGrayscale'),
        previewDenoised: document.getElementById('previewDenoised'),
        previewClahe: document.getElementById('previewClahe'),
        previewContour: document.getElementById('previewContour'),
        
        // Stage 3: Features
        feat_glcm_contrast: document.getElementById('feat_glcm_contrast'),
        feat_glcm_correlation: document.getElementById('feat_glcm_correlation'),
        feat_glcm_homogeneity: document.getElementById('feat_glcm_homogeneity'),
        feat_glcm_energy: document.getElementById('feat_glcm_energy'),
        feat_glcm_dissimilarity: document.getElementById('feat_glcm_dissimilarity'),
        feat_edge_gradient_mean: document.getElementById('feat_edge_gradient_mean'),
        feat_asymmetry_index: document.getElementById('feat_asymmetry_index'),
        feat_border_irregularity: document.getElementById('feat_border_irregularity'),
        feat_color_variegation: document.getElementById('feat_color_variegation'),
        feat_equivalent_diameter: document.getElementById('feat_equivalent_diameter'),
        feat_solidity: document.getElementById('feat_solidity'),
        feat_compactness: document.getElementById('feat_compactness'),
        radarSvg: document.getElementById('radarSvg'),
        
        // Stage 4: Training
        modelAlgSelect: document.getElementById('modelAlgSelect'),
        nTreesInput: document.getElementById('nTreesInput'),
        cParamInput: document.getElementById('cParamInput'),
        kNeighborsInput: document.getElementById('kNeighborsInput'),
        groupTrees: document.getElementById('groupTrees'),
        groupC: document.getElementById('groupC'),
        groupNeighbors: document.getElementById('groupNeighbors'),
        trainSplitSlider: document.getElementById('trainSplitSlider'),
        splitRatioVal: document.getElementById('splitRatioVal'),
        btnTrainModel: document.getElementById('btnTrainModel'),
        trainingStatusBadge: document.getElementById('trainingStatusBadge'),
        trainAccVal: document.getElementById('trainAccVal'),
        trainPrecVal: document.getElementById('trainPrecVal'),
        trainRecVal: document.getElementById('trainRecVal'),
        trainAucVal: document.getElementById('trainAucVal'),
        fiBarsList: document.getElementById('fiBarsList'),
        
        // Stage 5: Verdict
        verdictHeroCard: document.getElementById('verdictHeroCard'),
        verdictBadge: document.getElementById('verdictBadge'),
        riskBadge: document.getElementById('riskBadge'),
        verdictHeadline: document.getElementById('verdictHeadline'),
        verdictAdvice: document.getElementById('verdictAdvice'),
        verdictAlgLabel: document.getElementById('verdictAlgLabel'),
        verdictMorphologyLabel: document.getElementById('verdictMorphologyLabel'),
        gaugeCircleFill: document.getElementById('gaugeCircleFill'),
        gaugePercentVal: document.getElementById('gaugePercentVal'),
        val_tn: document.getElementById('val_tn'),
        val_fp: document.getElementById('val_fp'),
        val_fn: document.getElementById('val_fn'),
        val_tp: document.getElementById('val_tp'),
        rocAucText: document.getElementById('rocAucText'),
        rocSvg: document.getElementById('rocSvg'),
        btnPrintReport: document.getElementById('btnPrintReport'),
        
        // Code Sandbox Dock
        codeSandboxDock: document.getElementById('codeSandboxDock'),
        mainWorkspace: document.querySelector('.main-workspace'),
        codeTemplateSelect: document.getElementById('codeTemplateSelect'),
        btnCloseSandbox: document.getElementById('btnCloseSandbox'),
        codeEditor: document.getElementById('codeEditor'),
        lineNumbers: document.getElementById('lineNumbers'),
        btnRunCode: document.getElementById('btnRunCode'),
        btnResetCode: document.getElementById('btnResetCode'),
        btnCopyCode: document.getElementById('btnCopyCode'),
        terminalOutput: document.getElementById('terminalOutput'),
        execDurationText: document.getElementById('execDurationText'),
        
        // Modal Guide
        docsModal: document.getElementById('docsModal'),
        btnCloseDocsModal: document.getElementById('btnCloseDocsModal'),
        btnModalAcknowledge: document.getElementById('btnModalAcknowledge')
    };

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    async function initApp() {
        setupEventListeners();
        updateLineNumbers();
        drawEmptyRadar();
        drawRocCurve([
            { fpr: 0.0, tpr: 0.0 },
            { fpr: 0.0, tpr: 1.0 },
            { fpr: 1.0, tpr: 1.0 }
        ]);

        try {
            await fetchSystemStatus();
            await fetchCodeTemplates();
            await fetchBenchmarkSamples();
        } catch (err) {
            console.error('Initialization error:', err);
        }
    }

    // -------------------------------------------------------------------------
    // API Fetchers
    // -------------------------------------------------------------------------
    async function fetchSystemStatus() {
        const res = await fetch('/api/status');
        const data = await res.json();
        elements.engineStatus.textContent = 'Operational (Python 3.14)';
        elements.activeModelLabel.textContent = formatAlgName(data.active_algorithm);
        elements.datasetCountLabel.textContent = `${data.total_dataset_records} Patients`;
        state.activeAlgorithm = data.active_algorithm;
        elements.modelAlgSelect.value = data.active_algorithm;
        updateDynamicParamsUI(data.active_algorithm);
    }

    async function fetchBenchmarkSamples() {
        const res = await fetch('/api/samples');
        const data = await res.json();
        renderBenchmarkCards(data.samples);

        // Auto-select first sample to populate pipeline immediately
        if (data.samples && data.samples.length > 0) {
            loadBenchmarkSample(data.samples[0].id);
        }
    }

    async function fetchCodeTemplates() {
        const res = await fetch('/api/code-templates');
        const data = await res.json();
        state.codeTemplates = data.templates;
        if (state.codeTemplates['pipeline_demo']) {
            elements.codeEditor.value = state.codeTemplates['pipeline_demo'].code;
            updateLineNumbers();
        }
    }

    // -------------------------------------------------------------------------
    // Benchmark Samples Rendering
    // -------------------------------------------------------------------------
    function renderBenchmarkCards(samples) {
        elements.samplesContainer.innerHTML = '';
        samples.forEach(sample => {
            const card = document.createElement('div');
            card.className = `sample-thumb-card ${sample.id === state.currentSampleId ? 'active' : ''}`;
            card.dataset.id = sample.id;
            card.title = `${sample.title} — ${sample.description}`;

            const badgeClass = sample.is_malignant ? 'malignant' : 'benign';
            card.innerHTML = `
                <img src="${sample.url}" alt="${sample.title}" class="sample-thumb-img">
                <span class="sample-thumb-title">${sample.title}</span>
                <span class="sample-type-badge ${badgeClass}">${sample.category}</span>
            `;

            card.addEventListener('click', () => {
                document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                loadBenchmarkSample(sample.id);
            });

            elements.samplesContainer.appendChild(card);
        });
    }

    // -------------------------------------------------------------------------
    // Image Pipeline Execution
    // -------------------------------------------------------------------------
    async function loadBenchmarkSample(sampleId) {
        state.currentSampleId = sampleId;
        state.currentImageSource = 'sample';
        state.uploadedFile = null;

        setLoadingState(true);
        updateActiveStep(1);

        try {
            const payload = {
                sample_id: sampleId,
                filter_type: elements.filterTypeSelect.value,
                clahe_clip: parseFloat(elements.claheClipSlider.value),
                algorithm: state.activeAlgorithm
            };

            const res = await fetch('/api/pipeline/process-sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            handlePipelineSuccess(data);
        } catch (err) {
            alert(`Processing failed: ${err.message}`);
        } finally {
            setLoadingState(false);
        }
    }

    async function uploadCustomImage(file) {
        state.uploadedFile = file;
        state.currentImageSource = 'upload';
        state.currentSampleId = null;
        document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));

        setLoadingState(true);
        updateActiveStep(1);

        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('filter_type', elements.filterTypeSelect.value);
            formData.append('clahe_clip', elements.claheClipSlider.value);
            formData.append('algorithm', state.activeAlgorithm);

            const res = await fetch('/api/pipeline/upload', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            handlePipelineSuccess(data);
        } catch (err) {
            alert(`Upload failed: ${err.message}`);
        } finally {
            setLoadingState(false);
        }
    }

    function handlePipelineSuccess(data) {
        // Stage 1 Metadata
        elements.activeImageMetaBar.style.display = 'flex';
        elements.metaFileName.textContent = data.source;
        elements.metaDimensions.textContent = `${data.image_metadata.width} × ${data.image_metadata.height} px`;

        // Stage 2 Previews
        elements.previewOriginal.src = data.previews.original;
        elements.previewGrayscale.src = data.previews.grayscale;
        elements.previewDenoised.src = data.previews.denoised;
        elements.previewClahe.src = data.previews.clahe_enhanced;
        elements.previewContour.src = data.previews.contour_overlay;

        // Stage 3 Features
        populateFeatures(data.features);
        drawRadarChart(data.radar_metrics);

        // Stage 5 Verdict
        renderDiagnosticVerdict(data.prediction, data.features);

        // Mark steps complete
        markAllStepsComplete();
    }

    function populateFeatures(f) {
        elements.feat_glcm_contrast.textContent = f.glcm_contrast ?? '-';
        elements.feat_glcm_correlation.textContent = f.glcm_correlation ?? '-';
        elements.feat_glcm_homogeneity.textContent = f.glcm_homogeneity ?? '-';
        elements.feat_glcm_energy.textContent = f.glcm_energy ?? '-';
        elements.feat_glcm_dissimilarity.textContent = f.glcm_dissimilarity ?? '-';
        elements.feat_edge_gradient_mean.textContent = f.edge_gradient_mean ?? '-';

        elements.feat_asymmetry_index.textContent = f.asymmetry_index ?? '-';
        elements.feat_border_irregularity.textContent = f.border_irregularity ?? '-';
        elements.feat_color_variegation.textContent = f.color_variegation ?? '-';
        elements.feat_equivalent_diameter.textContent = f.equivalent_diameter ? `${f.equivalent_diameter} px` : '-';
        elements.feat_solidity.textContent = f.solidity ?? '-';
        elements.feat_compactness.textContent = f.compactness ?? '-';
    }

    function renderDiagnosticVerdict(pred, features) {
        const isMalignant = pred.is_malignant;
        
        elements.verdictHeroCard.classList.remove('malignant', 'benign');
        elements.verdictHeroCard.classList.add(isMalignant ? 'malignant' : 'benign');

        elements.verdictBadge.textContent = isMalignant ? 'MALIGNANT NEOPLASM' : 'BENIGN LESION';
        elements.riskBadge.textContent = `Risk: ${pred.risk_tier}`;
        elements.verdictHeadline.textContent = isMalignant 
            ? 'Malignancy Detected (High Diagnostic Probability)' 
            : 'Benign Cellular Morphology Confirmed';
        elements.verdictAdvice.textContent = pred.clinical_recommendation;
        elements.verdictAlgLabel.textContent = formatAlgName(pred.algorithm_used);
        
        // Morphological summary
        const asym = features.asymmetry_index >= 0.4 ? 'Asymmetrical' : 'Symmetrical';
        const border = features.border_irregularity >= 0.4 ? 'Irregular Border' : 'Smooth Border';
        elements.verdictMorphologyLabel.textContent = `${asym}, ${border}`;

        // Animate Confidence Gauge
        const percent = pred.confidence_percentage;
        elements.gaugePercentVal.textContent = `${percent}%`;

        const circumference = 314.16; // 2 * pi * 50
        const offset = circumference - (percent / 100) * circumference;
        elements.gaugeCircleFill.style.strokeDashoffset = offset;
    }

    // -------------------------------------------------------------------------
    // Model Training Studio
    // -------------------------------------------------------------------------
    async function trainModel() {
        if (state.isTraining) return;
        state.isTraining = true;
        elements.btnTrainModel.disabled = true;
        elements.btnTrainModel.innerHTML = '<span class="mini-spinner"></span> Training Model...';
        elements.trainingStatusBadge.textContent = 'Training in progress...';
        elements.trainingStatusBadge.style.color = '#f59e0b';

        const alg = elements.modelAlgSelect.value;
        const testSplit = parseFloat(elements.trainSplitSlider.value);

        const payload = {
            algorithm: alg,
            test_split: testSplit,
            n_estimators: parseInt(elements.nTreesInput.value, 10),
            c_param: parseFloat(elements.cParamInput.value),
            k_neighbors: parseInt(elements.kNeighborsInput.value, 10)
        };

        try {
            const res = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error(await res.text());
            const data = await res.json();
            const m = data.metrics;

            // Update training telemetry cards
            elements.trainAccVal.textContent = `${(m.accuracy * 100).toFixed(1)}%`;
            elements.trainPrecVal.textContent = m.precision.toFixed(4);
            elements.trainRecVal.textContent = m.recall.toFixed(4);
            elements.trainAucVal.textContent = m.roc_auc.toFixed(4);

            // Update confusion matrix
            const cm = m.confusion_matrix;
            elements.val_tn.textContent = cm.true_negative;
            elements.val_fp.textContent = cm.false_positive;
            elements.val_fn.textContent = cm.false_negative;
            elements.val_tp.textContent = cm.true_positive;

            // Update ROC Curve
            elements.rocAucText.textContent = m.roc_auc.toFixed(3);
            drawRocCurve(m.roc_curve);

            // Update Feature Importance Bars
            renderFeatureImportanceBars(m.feature_importances);

            // Update active algorithm state
            state.activeAlgorithm = alg;
            elements.activeModelLabel.textContent = formatAlgName(alg);
            elements.trainingStatusBadge.textContent = 'Model Cached & Validated';
            elements.trainingStatusBadge.style.color = '#10b981';

            // Re-predict current sample with newly trained model
            if (state.currentSampleId) {
                loadBenchmarkSample(state.currentSampleId);
            } else if (state.uploadedFile) {
                uploadCustomImage(state.uploadedFile);
            }
        } catch (err) {
            alert(`Training error: ${err.message}`);
            elements.trainingStatusBadge.textContent = 'Training Failed';
            elements.trainingStatusBadge.style.color = '#f43f5e';
        } finally {
            state.isTraining = false;
            elements.btnTrainModel.disabled = false;
            elements.btnTrainModel.innerHTML = '<span class="btn-icon">⚡</span> Train Classifier Now';
        }
    }

    function renderFeatureImportanceBars(importances) {
        elements.fiBarsList.innerHTML = '';
        if (!importances || importances.length === 0) {
            elements.fiBarsList.innerHTML = '<span class="fi-title">Feature importance not applicable for this model.</span>';
            return;
        }

        const maxImp = Math.max(...importances.map(i => i.importance), 0.001);

        importances.slice(0, 5).forEach(item => {
            const percent = Math.min((item.importance / maxImp) * 100, 100);
            const row = document.createElement('div');
            row.className = 'fi-bar-row';
            row.innerHTML = `
                <span class="fi-name" title="${item.name}">${formatFeatureName(item.name)}</span>
                <div class="fi-bar-track">
                    <div class="fi-bar-fill" style="width: ${percent}%;"></div>
                </div>
                <span class="fi-val">${item.importance.toFixed(3)}</span>
            `;
            elements.fiBarsList.appendChild(row);
        });
    }

    // -------------------------------------------------------------------------
    // In-Browser Interactive Python Code Compiler & Sandbox
    // -------------------------------------------------------------------------
    async function runCodeInSandbox() {
        const code = elements.codeEditor.value;
        if (!code.trim()) {
            alert('Editor is empty. Please type or select code to execute.');
            return;
        }

        elements.btnRunCode.disabled = true;
        elements.btnRunCode.innerHTML = '<span class="mini-spinner"></span> Running...';
        elements.terminalOutput.className = 'terminal-output';
        elements.terminalOutput.textContent = 'Compiling and executing code in Python 3.14 subprocess...\n';

        try {
            const res = await fetch('/api/run-code', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: code, timeout: 15 })
            });

            const data = await res.json();
            elements.execDurationText.textContent = `${data.duration_ms} ms`;

            if (data.success) {
                elements.terminalOutput.textContent = data.stdout || '(Program completed with return code 0, no stdout output)';
                elements.terminalOutput.className = 'terminal-output';
            } else {
                elements.terminalOutput.textContent = `${data.stdout}\n[ERROR]:\n${data.stderr}`;
                elements.terminalOutput.className = 'terminal-output error';
            }
        } catch (err) {
            elements.terminalOutput.textContent = `Server communication failure: ${err.message}`;
            elements.terminalOutput.className = 'terminal-output error';
        } finally {
            elements.btnRunCode.disabled = false;
            elements.btnRunCode.innerHTML = '<span class="btn-icon">▶</span> Run Program';
        }
    }

    // -------------------------------------------------------------------------
    // Dynamic SVG Radar & ROC Chart Visualizers
    // -------------------------------------------------------------------------
    function drawEmptyRadar() {
        const defaultMetrics = [
            { axis: 'Asymmetry (A)', value: 0.2, malignant_threshold: 0.45 },
            { axis: 'Border Irreg. (B)', value: 0.2, malignant_threshold: 0.40 },
            { axis: 'Color Varieg. (C)', value: 0.2, malignant_threshold: 0.50 },
            { axis: 'GLCM Contrast', value: 0.2, malignant_threshold: 0.45 },
            { axis: 'Edge Density', value: 0.2, malignant_threshold: 0.50 },
            { axis: 'Texture Entropy', value: 0.2, malignant_threshold: 0.45 }
        ];
        drawRadarChart(defaultMetrics);
    }

    function drawRadarChart(metrics) {
        const svg = elements.radarSvg;
        svg.innerHTML = '';

        const cx = 160, cy = 160, radius = 100;
        const totalAxes = metrics.length;
        const angleSlice = (Math.PI * 2) / totalAxes;

        // Draw concentric reference circles
        for (let level = 1; level <= 4; level++) {
            const r = (radius / 4) * level;
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', cx);
            circle.setAttribute('cy', cy);
            circle.setAttribute('r', r);
            circle.setAttribute('fill', 'none');
            circle.setAttribute('stroke', 'rgba(255, 255, 255, 0.08)');
            circle.setAttribute('stroke-dasharray', '2 3');
            svg.appendChild(circle);
        }

        // Draw Axes & Labels
        metrics.forEach((m, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            const x2 = cx + radius * Math.cos(angle);
            const y2 = cy + radius * Math.sin(angle);

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', cx);
            line.setAttribute('y1', cy);
            line.setAttribute('x2', x2);
            line.setAttribute('y2', y2);
            line.setAttribute('stroke', 'rgba(255, 255, 255, 0.12)');
            svg.appendChild(line);

            // Axis Label
            const lx = cx + (radius + 24) * Math.cos(angle);
            const ly = cy + (radius + 18) * Math.sin(angle);
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', lx);
            text.setAttribute('y', ly);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('dominant-baseline', 'middle');
            text.setAttribute('fill', '#94a3b8');
            text.setAttribute('font-size', '8px');
            text.setAttribute('font-family', 'Outfit, sans-serif');
            text.textContent = m.axis;
            svg.appendChild(text);
        });

        // 1. Draw Malignancy Threshold Polygon
        const threshPoints = metrics.map((m, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            const r = radius * m.malignant_threshold;
            return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
        }).join(' ');

        const threshPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        threshPoly.setAttribute('points', threshPoints);
        threshPoly.setAttribute('fill', 'rgba(244, 63, 94, 0.1)');
        threshPoly.setAttribute('stroke', '#f43f5e');
        threshPoly.setAttribute('stroke-width', '1.5');
        threshPoly.setAttribute('stroke-dasharray', '4 3');
        svg.appendChild(threshPoly);

        // 2. Draw Current Specimen Polygon
        const specimenPoints = metrics.map((m, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            const val = Math.min(Math.max(m.value, 0.05), 1.0);
            const r = radius * val;
            return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
        }).join(' ');

        const specPoly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        specPoly.setAttribute('points', specimenPoints);
        specPoly.setAttribute('fill', 'rgba(6, 182, 212, 0.35)');
        specPoly.setAttribute('stroke', '#06b6d4');
        specPoly.setAttribute('stroke-width', '2.5');
        svg.appendChild(specPoly);

        // Draw Vertex Points
        metrics.forEach((m, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            const val = Math.min(Math.max(m.value, 0.05), 1.0);
            const px = cx + (radius * val) * Math.cos(angle);
            const py = cy + (radius * val) * Math.sin(angle);

            const pt = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            pt.setAttribute('cx', px);
            pt.setAttribute('cy', py);
            pt.setAttribute('r', 4);
            pt.setAttribute('fill', '#22d3ee');
            pt.setAttribute('stroke', '#083344');
            pt.setAttribute('stroke-width', '1.5');
            svg.appendChild(pt);
        });
    }

    function drawRocCurve(points) {
        const svg = elements.rocSvg;
        svg.innerHTML = '';

        const padLeft = 35, padBottom = 25, width = 240, height = 180;

        // Axes
        const axes = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        axes.setAttribute('d', `M ${padLeft} 10 L ${padLeft} ${height + 10} L ${padLeft + width} ${height + 10}`);
        axes.setAttribute('stroke', 'rgba(255, 255, 255, 0.15)');
        axes.setAttribute('stroke-width', '1');
        axes.setAttribute('fill', 'none');
        svg.appendChild(axes);

        // Diagonal Chance Line (FPR = TPR)
        const diag = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        diag.setAttribute('x1', padLeft);
        diag.setAttribute('y1', height + 10);
        diag.setAttribute('x2', padLeft + width);
        diag.setAttribute('y2', 10);
        diag.setAttribute('stroke', 'rgba(255, 255, 255, 0.2)');
        diag.setAttribute('stroke-dasharray', '4 4');
        svg.appendChild(diag);

        // ROC Curve Line
        if (points && points.length > 0) {
            let pathD = '';
            points.forEach((pt, idx) => {
                const x = padLeft + pt.fpr * width;
                const y = (height + 10) - pt.tpr * height;
                if (idx === 0) pathD += `M ${x} ${y} `;
                else pathD += `L ${x} ${y} `;
            });

            const rocPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            rocPath.setAttribute('d', pathD);
            rocPath.setAttribute('stroke', '#06b6d4');
            rocPath.setAttribute('stroke-width', '2.5');
            rocPath.setAttribute('fill', 'none');
            svg.appendChild(rocPath);
        }
    }

    // -------------------------------------------------------------------------
    // Event Listeners & UI Helpers
    // -------------------------------------------------------------------------
    function setupEventListeners() {
        // Drag & Drop
        const dropZone = elements.dropZone;
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, e => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            });
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, e => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            });
        });
        dropZone.addEventListener('drop', e => {
            if (e.dataTransfer.files.length > 0) {
                uploadCustomImage(e.dataTransfer.files[0]);
            }
        });
        dropZone.addEventListener('click', () => elements.imageInput.click());
        elements.imageInput.addEventListener('change', e => {
            if (e.target.files.length > 0) {
                uploadCustomImage(e.target.files[0]);
            }
        });

        // Stage 2 Filters
        elements.claheClipSlider.addEventListener('input', e => {
            elements.claheVal.textContent = parseFloat(e.target.value).toFixed(1);
        });
        elements.btnReProcess.addEventListener('click', () => {
            if (state.currentSampleId) loadBenchmarkSample(state.currentSampleId);
            else if (state.uploadedFile) uploadCustomImage(state.uploadedFile);
        });

        // Stage 4 Training controls
        elements.modelAlgSelect.addEventListener('change', e => {
            updateDynamicParamsUI(e.target.value);
        });
        elements.trainSplitSlider.addEventListener('input', e => {
            const split = Math.round(parseFloat(e.target.value) * 100);
            elements.splitRatioVal.textContent = `${split}% (${100 - split}/${split} Split)`;
        });
        elements.btnTrainModel.addEventListener('click', trainModel);

        // Header Actions
        elements.btnRunFullPipeline.addEventListener('click', () => {
            if (state.currentSampleId) loadBenchmarkSample(state.currentSampleId);
            else {
                const firstCard = document.querySelector('.sample-thumb-card');
                if (firstCard) firstCard.click();
            }
        });

        // Code Sandbox Toggle
        elements.btnToggleSandbox.addEventListener('click', toggleCodeSandbox);
        elements.btnCloseSandbox.addEventListener('click', toggleCodeSandbox);

        // Code Editor
        elements.codeTemplateSelect.addEventListener('change', e => {
            const tmpl = state.codeTemplates[e.target.value];
            if (tmpl) {
                elements.codeEditor.value = tmpl.code;
                updateLineNumbers();
            }
        });
        elements.codeEditor.addEventListener('input', updateLineNumbers);
        elements.codeEditor.addEventListener('scroll', () => {
            elements.lineNumbers.scrollTop = elements.codeEditor.scrollTop;
        });
        elements.btnRunCode.addEventListener('click', runCodeInSandbox);
        elements.btnResetCode.addEventListener('click', () => {
            const key = elements.codeTemplateSelect.value;
            if (state.codeTemplates[key]) {
                elements.codeEditor.value = state.codeTemplates[key].code;
                updateLineNumbers();
            }
        });
        elements.btnCopyCode.addEventListener('click', () => {
            navigator.clipboard.writeText(elements.codeEditor.value);
            elements.btnCopyCode.textContent = 'Copied!';
            setTimeout(() => { elements.btnCopyCode.textContent = 'Copy Code'; }, 1500);
        });

        // Docs Modal
        elements.btnOpenDocsModal.addEventListener('click', () => { elements.docsModal.style.display = 'flex'; });
        elements.btnCloseDocsModal.addEventListener('click', () => { elements.docsModal.style.display = 'none'; });
        elements.btnModalAcknowledge.addEventListener('click', () => { elements.docsModal.style.display = 'none'; });
        elements.docsModal.addEventListener('click', e => {
            if (e.target === elements.docsModal) elements.docsModal.style.display = 'none';
        });

        // Snippet copy buttons
        document.querySelectorAll('.btn-copy-snippet').forEach(btn => {
            btn.addEventListener('click', () => {
                const text = btn.dataset.copy;
                navigator.clipboard.writeText(text);
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            });
        });

        // Print Report
        elements.btnPrintReport.addEventListener('click', () => window.print());

        // Stepper Navigation
        elements.stepItems.forEach(item => {
            item.addEventListener('click', () => {
                const step = item.dataset.step;
                const targetCard = document.getElementById(`stage${step}-card`);
                if (targetCard) targetCard.scrollIntoView({ behavior: 'smooth' });
                updateActiveStep(step);
            });
        });
    }

    function toggleCodeSandbox() {
        const isClosed = elements.mainWorkspace.classList.toggle('sandbox-closed');
        elements.codeSandboxDock.classList.toggle('open');
        elements.btnToggleSandbox.classList.toggle('active', !isClosed);
    }

    function updateDynamicParamsUI(alg) {
        elements.groupTrees.style.display = (alg === 'random_forest' || alg === 'gradient_boosting') ? 'flex' : 'none';
        elements.groupC.style.display = (alg === 'svm' || alg === 'logistic_regression') ? 'flex' : 'none';
        elements.groupNeighbors.style.display = (alg === 'knn') ? 'flex' : 'none';
    }

    function updateLineNumbers() {
        const lines = elements.codeEditor.value.split('\n').length;
        let numbersHtml = '';
        for (let i = 1; i <= lines; i++) {
            numbersHtml += `${i}<br>`;
        }
        elements.lineNumbers.innerHTML = numbersHtml;
    }

    function updateActiveStep(stepNum) {
        elements.stepItems.forEach(item => {
            const s = parseInt(item.dataset.step, 10);
            item.classList.toggle('active', s === parseInt(stepNum, 10));
        });
    }

    function markAllStepsComplete() {
        elements.stepItems.forEach(item => item.classList.add('completed'));
    }

    function setLoadingState(isLoading) {
        state.isProcessing = isLoading;
        document.querySelectorAll('.img-overlay-loading').forEach(el => {
            el.style.display = isLoading ? 'flex' : 'none';
        });
    }

    function formatAlgName(alg) {
        const map = {
            'random_forest': 'Random Forest',
            'svm': 'Support Vector Machine (SVM)',
            'knn': 'K-Nearest Neighbors (k-NN)',
            'logistic_regression': 'Logistic Regression',
            'gradient_boosting': 'Gradient Boosting Trees'
        };
        return map[alg] || alg;
    }

    function formatFeatureName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Run Init
    initApp();
});
