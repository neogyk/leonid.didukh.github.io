import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { pipeline, env } from '@xenova/transformers';

// --- Configuration ---
env.allowLocalModels = false;
env.useBrowserCache = true;

const CONFIG = {
    pointRadius: 0.1,
    pointColor: 0x0066cc, // Darker blue for visibility on white
    selectedColor: 0xff0055, // Vivid red/pink
    lineColor: 0x333333, // Dark grey lines
    lineOpacity: 0.4,
    embeddingScale: 5, // t-SNE output needs less scaling than our pseudo-random one
    arrowColor: 0xff8800, // Orange arrows
    arrowLength: 0.8,
    colors: {
        combined: 0x0066cc,
        problem: 0x0066cc, // Blue
        method: 0x00cc66, // Green
        conclusion: 0xff8800 // Orange
    }
};

// Global State
let globalRawData = [];
let embeddingCache = {}; // Cache: text -> embedding array
let currentMode = 'triplets'; // 'triplets' or 'components'

// --- Helper Functions ---

// Data Loading Helper
function readTextFile(file, callback) {
    var rawFile = new XMLHttpRequest();
    rawFile.overrideMimeType("application/json");
    rawFile.open("GET", file, true);
    rawFile.onreadystatechange = function() {
        if (rawFile.readyState === 4 && rawFile.status == "200") {
            callback(rawFile.responseText);
        }
    }
    rawFile.send(null);
}

// UI Helper for Loading
function updateLoadingStatus(message) {
    const tooltip = document.getElementById('tooltip');
    if (message) {
        tooltip.innerHTML = message;
        tooltip.classList.remove('hidden');
        tooltip.style.left = '50%';
        tooltip.style.top = '50%';
        tooltip.style.transform = 'translate(-50%, -50%)';
        tooltip.style.zIndex = '1000';
    } else {
        tooltip.classList.add('hidden');
        tooltip.style.transform = 'translate(-50%, -100%)'; // Reset to default tooltip behavior
        tooltip.style.left = '0';
        tooltip.style.top = '0';
        tooltip.style.zIndex = '10';
    }
}

// Embedding Generation
let featureExtractor = null;

async function generateEmbeddings(texts) {
    try {
        if (!featureExtractor) {
            updateLoadingStatus("Loading embedding model (Xenova/all-MiniLM-L6-v2)...<br>This happens in your browser and may take a moment.");
            featureExtractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        }

        updateLoadingStatus("Generating embeddings...");
        const embeddings = [];
        
        for (let i = 0; i < texts.length; i++) {
            const text = texts[i];
            
            // Check cache
            if (embeddingCache[text]) {
                embeddings.push(embeddingCache[text]);
                continue;
            }

            // Update progress occasionally
            updateLoadingStatus(`Generating embeddings: ${i + 1}/${texts.length}`);
            
            // Generate embedding
            const output = await featureExtractor(text, { pooling: 'mean', normalize: true });
            const vector = Array.from(output.data);
            
            // Cache it
            embeddingCache[text] = vector;
            embeddings.push(vector);
        }
        
        return embeddings;
    } catch (e) {
        console.error("Embedding generation failed:", e);
        alert("Failed to load model or generate embeddings. See console.");
        return [];
    }
}

// t-SNE Projection
async function runTSNE(embeddings) {
    updateLoadingStatus("Running t-SNE projection...");
    
    return new Promise((resolve) => {
        setTimeout(() => {
            try {
                // Initialize t-SNE
                const opt = {
                    epsilon: 10, 
                    perplexity: Math.max(5, Math.min(30, embeddings.length - 1)), 
                    dim: 3 
                };
                
                // tsnejs usually exposes 'tsne' object with 'tSNE' constructor
                const T = new window.tsne.tSNE(opt); 
                T.initDataRaw(embeddings);
                
                // Run optimization
                const iterations = 500;
                for (let k = 0; k < iterations; k++) {
                    T.step();
                }
                
                resolve(T.getSolution()); // Returns array of [x, y, z]
            } catch (e) {
                console.error("t-SNE failed:", e);
                // Fallback to random if t-SNE fails
                resolve(embeddings.map(() => [Math.random(), Math.random(), Math.random()]));
            }
        }, 100); // Small delay to let UI render the message
    });
}


// --- Scene Setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0); // White/Light Grey background
scene.fog = new THREE.FogExp2(0xf0f0f0, 0.02);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 5, 20);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.getElementById('container').appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- Lighting ---
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const pointLight = new THREE.PointLight(0xffffff, 1);
pointLight.position.set(10, 10, 10);
scene.add(pointLight);

// --- Visualization State ---
let interactableObjects = []; 
let selectedObject = null;
let hoveredObject = null;

// --- Build Visualization ---
function buildScene(trajectoryData) {
    // Clear previous scene elements
    while(scene.children.length > 0){ 
        scene.remove(scene.children[0]); 
    }
    // Re-add lights
    scene.add(ambientLight);
    scene.add(pointLight);
    
    interactableObjects = [];

    const nodeGeometry = new THREE.SphereGeometry(CONFIG.pointRadius, 32, 32);
    const lineMaterial = new THREE.LineBasicMaterial({ 
        color: CONFIG.lineColor, 
        transparent: true, 
        opacity: CONFIG.lineOpacity 
    });

    const points = [];
    const group = new THREE.Group();
    scene.add(group);

    trajectoryData.forEach((step) => {
        const { x, y, z } = step.position;
        
        let color = CONFIG.pointColor;
        if (step.type && CONFIG.colors[step.type]) {
            color = CONFIG.colors[step.type];
        }

        const nodeMaterial = new THREE.MeshStandardMaterial({ 
            color: color,
            roughness: 0.3,
            metalness: 0.5,
            emissive: 0x000000 
        });

        const mesh = new THREE.Mesh(nodeGeometry, nodeMaterial);
        mesh.position.set(x, y, z);
        mesh.userData = { ...step, originalColor: color }; 
        
        group.add(mesh);
        interactableObjects.push(mesh);
        points.push(new THREE.Vector3(x, y, z));
    });

    // Draw lines and arrows
    if (points.length > 1) {
        const curve = new THREE.CatmullRomCurve3(points);
        const pointsCount = points.length * 10;
        const curvePoints = curve.getPoints(pointsCount);
        const curveGeometry = new THREE.BufferGeometry().setFromPoints(curvePoints);
        const curveLine = new THREE.Line(curveGeometry, lineMaterial);
        group.add(curveLine);

        for (let i = 0; i < points.length - 1; i++) {
            const startPoint = points[i];
            const endPoint = points[i + 1];
            
            const direction = new THREE.Vector3().subVectors(endPoint, startPoint).normalize();
            
            // Place arrow in the middle of the segment
            const midPoint = new THREE.Vector3().addVectors(startPoint, endPoint).multiplyScalar(0.5);
            const distance = startPoint.distanceTo(endPoint);
            
            // Only draw if segment is long enough
            if (distance > CONFIG.pointRadius * 2.5) {
                // Adjust length to be small and centered
                const arrowLength = Math.min(CONFIG.arrowLength, distance * 0.4); 
                const arrowStart = midPoint.clone().sub(direction.clone().multiplyScalar(arrowLength / 2));

                const headLength = arrowLength * 0.4;
                const headWidth = headLength * 0.6;

                const arrowHelper = new THREE.ArrowHelper(
                    direction, 
                    arrowStart, 
                    arrowLength, 
                    CONFIG.arrowColor, 
                    headLength, 
                    headWidth
                );
                group.add(arrowHelper);
            }
        }
    }
    
    // Center camera on the centroid
    if (points.length > 0) {
        const center = new THREE.Vector3();
        points.forEach(p => center.add(p));
        center.divideScalar(points.length);
        controls.target.copy(center);
        // Do not reset camera position drastically if we are already viewing
        // camera.lookAt(center); 
    }

    // Add AxesHelper
    const axesHelper = new THREE.AxesHelper(10);
    scene.add(axesHelper);
}

async function processData() {
    try {
        let texts = [];
        let metadata = [];

        if (currentMode === 'triplets') {
            texts = globalRawData.map(step => `${step.p}\n${step.s}\n${step.c}`);
            metadata = globalRawData.map((step, index) => ({
                id: `step-${index}`,
                problem: step.p,
                method: step.s,
                conclusion: step.c,
                type: 'combined',
                label: `Step ${index + 1}`
            }));
        } else if (currentMode === 'components') {
            // Components mode: Unroll all
            globalRawData.forEach((step, index) => {
                // Problem
                texts.push(step.p);
                metadata.push({
                    id: `step-${index}-p`,
                    problem: step.p,
                    method: "(N/A - Problem View)",
                    conclusion: "(N/A - Problem View)",
                    type: 'problem',
                    label: `Step ${index + 1} (Problem)`
                });

                // Method
                texts.push(step.s);
                metadata.push({
                    id: `step-${index}-s`,
                    problem: "(N/A - Method View)",
                    method: step.s,
                    conclusion: "(N/A - Method View)",
                    type: 'method',
                    label: `Step ${index + 1} (Method)`
                });

                // Conclusion
                texts.push(step.c);
                metadata.push({
                    id: `step-${index}-c`,
                    problem: "(N/A - Conclusion View)",
                    method: "(N/A - Conclusion View)",
                    conclusion: step.c,
                    type: 'conclusion',
                    label: `Step ${index + 1} (Conclusion)`
                });
            });
        } else if (currentMode === 'problems') {
            globalRawData.forEach((step, index) => {
                texts.push(step.p);
                metadata.push({
                    id: `step-${index}-p`,
                    problem: step.p,
                    method: "---",
                    conclusion: "---",
                    type: 'problem',
                    label: `Step ${index + 1} (Problem Only)`
                });
            });
        } else if (currentMode === 'methods') {
            globalRawData.forEach((step, index) => {
                texts.push(step.s);
                metadata.push({
                    id: `step-${index}-s`,
                    problem: "---",
                    method: step.s,
                    conclusion: "---",
                    type: 'method',
                    label: `Step ${index + 1} (Method Only)`
                });
            });
        } else if (currentMode === 'conclusions') {
            globalRawData.forEach((step, index) => {
                texts.push(step.c);
                metadata.push({
                    id: `step-${index}-c`,
                    problem: "---",
                    method: "---",
                    conclusion: step.c,
                    type: 'conclusion',
                    label: `Step ${index + 1} (Conclusion Only)`
                });
            });
        }

        // 1. Generate Embeddings
        const embeddings = await generateEmbeddings(texts);
        
        // 2. Project with t-SNE
        const projectedPoints = await runTSNE(embeddings);
        
        updateLoadingStatus(""); // Clear loading message

        // 3. Process data for scene
        const trajectoryData = metadata.map((meta, index) => {
            const coords = projectedPoints[index];
            return {
                ...meta,
                position: new THREE.Vector3(
                    coords[0] * CONFIG.embeddingScale, 
                    coords[1] * CONFIG.embeddingScale, 
                    coords[2] * CONFIG.embeddingScale
                )
            };
        });

        buildScene(trajectoryData);

    } catch (e) {
        console.error("Error processing data:", e);
        alert("Error processing data.");
        updateLoadingStatus(""); 
    }
}

// --- Main execution ---
function init() {
    readTextFile("/api/data", function(text) {
        try {
            globalRawData = JSON.parse(text);
            processData(); // Initial load with default mode
        } catch (e) {
            console.error("Failed to parse initial data", e);
        }
    });
}

init();

// --- UI Logic ---
document.getElementById('viz-mode').addEventListener('change', (e) => {
    currentMode = e.target.value;
    processData();
});


// --- Interaction ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const infoPanel = document.getElementById('info-panel');
const tooltip = document.getElementById('tooltip');
const closeBtn = document.getElementById('close-panel');

const uiProblem = document.getElementById('point-problem');
const uiMethod = document.getElementById('point-method');
const uiConclusion = document.getElementById('point-conclusion');
const metadataPanel = document.getElementById('metadata-panel');

function updateInfoPanel(data) {
    if (data.type === 'combined') {
        uiProblem.textContent = data.problem;
        uiMethod.textContent = data.method;
        uiConclusion.textContent = data.conclusion;
    } else {
        // Customize panel based on type
        if (data.type === 'problem') {
            uiProblem.textContent = data.problem;
            uiMethod.textContent = "---";
            uiConclusion.textContent = "---";
        } else if (data.type === 'method') {
            uiProblem.textContent = "---";
            uiMethod.textContent = data.method;
            uiConclusion.textContent = "---";
        } else if (data.type === 'conclusion') {
            uiProblem.textContent = "---";
            uiMethod.textContent = "---";
            uiConclusion.textContent = data.conclusion;
        }
    }
    document.getElementById('point-title').textContent = data.label || 'Point Details';
    infoPanel.classList.remove('hidden');
}

function updateMetadataPanel(data) {
    document.getElementById('metadata-id').textContent = data.id || '-';
    document.getElementById('metadata-label').textContent = data.label || '-';
    document.getElementById('metadata-type').textContent = data.type || '-';
    
    // Calculate lengths
    const problemLength = data.problem ? data.problem.length : 0;
    const methodLength = data.method ? data.method.length : 0;
    const conclusionLength = data.conclusion ? data.conclusion.length : 0;
    
    document.getElementById('metadata-problem-length').textContent = `${problemLength} chars`;
    document.getElementById('metadata-method-length').textContent = `${methodLength} chars`;
    document.getElementById('metadata-conclusion-length').textContent = `${conclusionLength} chars`;
    
    // Calculate total words
    const problemWords = data.problem ? data.problem.split(/\s+/).filter(w => w.length > 0).length : 0;
    const methodWords = data.method ? data.method.split(/\s+/).filter(w => w.length > 0).length : 0;
    const conclusionWords = data.conclusion ? data.conclusion.split(/\s+/).filter(w => w.length > 0).length : 0;
    const totalWords = problemWords + methodWords + conclusionWords;
    
    document.getElementById('metadata-total-words').textContent = totalWords;
    
    metadataPanel.classList.remove('hidden');
}

function hideMetadataPanel() {
    metadataPanel.classList.add('hidden');
}

function resetSelection() {
    if (selectedObject) {
        selectedObject.material.color.setHex(selectedObject.userData.originalColor);
        selectedObject.scale.set(1, 1, 1);
        selectedObject = null;
    }
}

closeBtn.addEventListener('click', () => {
    infoPanel.classList.add('hidden');
    resetSelection();
});

function onMouseClick(event) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(interactableObjects);

    if (intersects.length > 0) {
        const object = intersects[0].object;
        if (selectedObject !== object) {
            resetSelection();
            selectedObject = object;
            selectedObject.material.color.setHex(CONFIG.selectedColor);
            selectedObject.scale.set(1.5, 1.5, 1.5);
            updateInfoPanel(object.userData);
        }
    }
}

window.addEventListener('click', onMouseClick);
window.addEventListener('mousemove', onMouseMove);

function onMouseMove(event) {
    // Only update hover if not showing global loading message
    if (!tooltip.innerHTML.includes("Loading") && !tooltip.innerHTML.includes("Generating")) {
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

        raycaster.setFromCamera(mouse, camera);
        const intersects = raycaster.intersectObjects(interactableObjects);

        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (hoveredObject !== object) {
                hoveredObject = object;
                tooltip.classList.remove('hidden');
                
                // Update metadata panel on hover
                updateMetadataPanel(object.userData);
                
                // Determine text to show based on type
                let displayText = "";
                if (object.userData.type === 'problem') displayText = object.userData.problem;
                else if (object.userData.type === 'method') displayText = object.userData.method;
                else if (object.userData.type === 'conclusion') displayText = object.userData.conclusion;
                else displayText = object.userData.problem; // Fallback for combined

                const textShort = displayText.length > 50 ? displayText.substring(0, 50) + "..." : displayText;
                const label = object.userData.label ? `<strong>${object.userData.label}</strong><br>` : '';
                tooltip.innerHTML = `${label}${textShort}`;
            }
            tooltip.style.left = event.clientX + 'px';
            tooltip.style.top = event.clientY + 'px';
            tooltip.style.transform = 'translate(-50%, -100%)';
        } else {
            if (hoveredObject) {
                hoveredObject = null;
                tooltip.classList.add('hidden');
                hideMetadataPanel();
            }
        }
    }
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

animate();