document.addEventListener('DOMContentLoaded', function() {
    // UI Elements
    const generateBtn = document.getElementById('generate-btn');
    const saveBtn = document.getElementById('save-btn');
    const exportBtn = document.getElementById('export-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const statusText = document.getElementById('status-text');
    const statusIndicator = document.querySelector('.status-indicator');
    const moleculeVisual = document.getElementById('molecule-visual');
    const resultsTable = document.getElementById('results-table');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const pageIndicator = document.getElementById('page-indicator');
    
    // Sliders
    const temperatureSlider = document.getElementById('temperature');
    const uniquenessSlider = document.getElementById('uniqueness');
    const complexitySlider = document.getElementById('complexity');
    
    // Property constraints
    const mwMinInput = document.getElementById('mw-min');
    const mwMaxInput = document.getElementById('mw-max');
    const logpMinInput = document.getElementById('logp-min');
    const logpMaxInput = document.getElementById('logp-max');
    
    // Molecule details elements
    const formulaElement = document.getElementById('formula');
    const molWeightElement = document.getElementById('mol-weight');
    const logpValueElement = document.getElementById('logp-value');
    const hDonorsElement = document.getElementById('h-donors');
    const hAcceptorsElement = document.getElementById('h-acceptors');
    const rotBondsElement = document.getElementById('rot-bonds');
    const validityScoreElement = document.getElementById('validity-score');
    const synthAccessElement = document.getElementById('sa-score'); // Matches HTML ID
    const drugLikenessElement = document.getElementById('qed'); // Updated ID
    
    // Modal elements
    const simulationModal = document.getElementById('simulation-modal');
    const simulationDisplay = document.getElementById('simulation-display');
    const closeModal = document.querySelector('.modal .close');
    
    // Model selection
    const modelSelect = document.getElementById('model');
    
    // Current page for pagination
    let currentPage = 1;
    let totalPages = 1;
    
    // Store molecules list for pagination
    let moleculesList = [];
    
    // Current selected molecule
    let currentMolecule = null;


    // Verify button exists
    if (!analyzeBtn) {
        console.error('Analyze button not found. Check ID: analyze-btn');
    }
    
    // Generate molecules on button click
    generateBtn.addEventListener('click', function() {
        statusText.textContent = 'Generating...';
        statusIndicator.classList.remove('ready', 'error');
        statusIndicator.classList.add('processing');
        
        const payload = {
            model: modelSelect.value,
            temperature: parseFloat(temperatureSlider.value),
            uniqueness: parseFloat(uniquenessSlider.value),
            complexity: parseFloat(complexitySlider.value),
            constraints: {
                mw_min: mwMinInput.value ? parseFloat(mwMinInput.value) : null,
                mw_max: mwMaxInput.value ? parseFloat(mwMaxInput.value) : null,
                logp_min: logpMinInput.value ? parseFloat(logpMinInput.value) : null,
                logp_max: logpMaxInput.value ? parseFloat(logpMaxInput.value) : null
            }
        };
        
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                moleculeVisual.innerHTML = `<img src="data:image/png;base64,${data.molecule.image}" alt="Generated Molecule">`;
                
                currentMolecule = data.molecule;
                formulaElement.textContent = data.molecule.formula || '-';
                molWeightElement.textContent = data.molecule.molWeight || '-';
                logpValueElement.textContent = data.molecule.logP || '-';
                hDonorsElement.textContent = data.molecule.h_donors || '-';
                hAcceptorsElement.textContent = data.molecule.h_acceptors || '-';
                rotBondsElement.textContent = data.molecule.rot_bonds || '-';
                validityScoreElement.textContent = data.molecule.validity_score || '-';
                synthAccessElement.textContent = data.molecule.synthetic_accessibility || '-'; // Updated key
                drugLikenessElement.textContent = data.molecule.drug_likeness || '-'; // Updated key
                
                moleculesList = data.molecules_list.map(mol => ({
                    ...mol,
                    image: mol.image,
                    large_image: mol.large_image,
                    formula: mol.formula,
                    molWeight: mol.molWeight,
                    logP: mol.logP,
                    validity: mol.validity,
                    uniqueness: mol.uniqueness,
                    h_donors: mol.h_donors || '-',
                    h_acceptors: mol.h_acceptors || '-',
                    rot_bonds: mol.rot_bonds || '-',
                    synthetic_accessibility: mol.synthetic_accessibility || '-', // Updated key
                    drug_likeness: mol.drug_likeness || '-' // Updated key
                }));
                // Store molecules in localStorage for Analysis page
                localStorage.setItem('molecules', JSON.stringify(moleculesList));
                
                updateResultsTable(moleculesList);
                
                statusText.textContent = 'Ready';
                statusIndicator.classList.remove('processing', 'error');
                statusIndicator.classList.add('ready');
                
                currentPage = 1;
                totalPages = Math.ceil(moleculesList.length / 5);
                updatePagination();
            } else {
                statusText.textContent = `Error: ${data.error}`;
                statusIndicator.classList.remove('processing', 'ready');
                statusIndicator.classList.add('error');
                moleculeVisual.innerHTML = `<div class="loading-placeholder">Error: ${data.error}</div>`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusText.textContent = 'Error connecting to server';
            statusIndicator.classList.remove('processing', 'ready');
            statusIndicator.classList.add('error');
            moleculeVisual.innerHTML = `<div class="loading-placeholder">Error connecting to server</div>`;
        });
    });
    
    function updateResultsTable(molecules) {
        resultsTable.innerHTML = '';
        const itemsPerPage = 5;
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = Math.min(startIndex + itemsPerPage, molecules.length);
        
        for (let i = startIndex; i < endIndex; i++) {
            const molecule = molecules[i];
            const row = document.createElement('tr');
            
            row.innerHTML = `
                <td><img src="data:image/png;base64,${molecule.image}" alt="Molecule" width="60" height="60"></td>
                <td>${molecule.formula}</td>
                <td>${molecule.molWeight}</td>
                <td>${molecule.logP}</td>
                <td>${molecule.validity}</td>
                <td>${molecule.uniqueness}</td>
                <td>
                    <div class="table-actions">
                        <button class="table-btn primary view-btn" data-index="${i}">View</button>
                        <button class="table-btn secondary simulate-btn" data-index="${i}">Simulate</button>
                    </div>
                </td>
            `;
            
            resultsTable.appendChild(row);
        }
        
        document.querySelectorAll('.view-btn').forEach(button => {
            button.addEventListener('click', function() {
                const index = parseInt(this.getAttribute('data-index'));
                viewMolecule(index);
            });
        });
        
        document.querySelectorAll('.simulate-btn').forEach(button => {
            button.addEventListener('click', function() {
                const index = parseInt(this.getAttribute('data-index'));
                simulateMolecule(index);
            });
        });
    }
    
    function updatePagination() {
        pageIndicator.textContent = `${currentPage} / ${totalPages}`;
        prevPageBtn.disabled = currentPage <= 1;
        nextPageBtn.disabled = currentPage >= totalPages;
        prevPageBtn.style.opacity = prevPageBtn.disabled ? '0.5' : '1';
        nextPageBtn.style.opacity = nextPageBtn.disabled ? '0.5' : '1';
    }
    
    function viewMolecule(index) {
        const molecule = moleculesList[index];
        currentMolecule = molecule;
        moleculeVisual.innerHTML = `<img src="data:image/png;base64,${molecule.large_image}" alt="Generated Molecule">`;
        formulaElement.textContent = molecule.formula || '-';
        molWeightElement.textContent = molecule.molWeight || '-';
        logpValueElement.textContent = molecule.logP || '-';
        hDonorsElement.textContent = molecule.h_donors || '-';
        hAcceptorsElement.textContent = molecule.h_acceptors || '-';
        rotBondsElement.textContent = molecule.rot_bonds || '-';
        validityScoreElement.textContent = molecule.validity || '-';
synthAccessElement.textContent = molecule.synthetic_accessibility || '-'; // Updated key
        drugLikenessElement.textContent = molecule.drug_likeness || '-';
    }
    
    function simulateMolecule(index) {
        const molecule = moleculesList[index];
        statusText.textContent = 'Simulating...';
        statusIndicator.classList.remove('ready', 'error');
        statusIndicator.classList.add('processing');
        
        fetch('/simulate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                smiles: molecule.smiles,
                temperature: parseFloat(temperatureSlider.value),
                seed: index
            })
        })
        .then(response => response.json())
        .then(data => {
            statusText.textContent = 'Ready';
            statusIndicator.classList.remove('processing', 'error');
            statusIndicator.classList.add('ready');
            
            if (data.success) {
                simulationDisplay.innerHTML = `<img src="data:image/gif;base64,${data.animation}" alt="Molecule Simulation">`;
                simulationModal.style.display = 'block';
            } else {
                alert(`Simulation failed: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Error simulating molecule:', error);
            statusText.textContent = 'Error';
            statusIndicator.classList.remove('processing', 'ready');
            statusIndicator.classList.add('error');
            alert('Error simulating molecule.');
        });
    }
    
    prevPageBtn.addEventListener('click', function() {
        if (currentPage > 1) {
            currentPage--;
            updateResultsTable(moleculesList);
            updatePagination();
        }
    });
    
    nextPageBtn.addEventListener('click', function() {
        if (currentPage < totalPages) {
            currentPage++;
            updateResultsTable(moleculesList);
            updatePagination();
        }
    });
    
    saveBtn.addEventListener('click', function() {
        if (!currentMolecule) {
            alert('No molecule selected to save.');
            return;
        }
        fetch('/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ molecule: currentMolecule })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Molecule saved successfully!');
            } else {
                alert('Failed to save molecule.');
            }
        })
        .catch(error => {
            console.error('Error saving molecule:', error);
            alert('Error saving molecule.');
        });
    });
    
    exportBtn.addEventListener('click', function() {
        if (!currentMolecule) {
            alert('No molecule selected to export.');
            return;
        }
        const dataStr = `data:text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(currentMolecule))}`;
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute('href', dataStr);
        downloadAnchor.setAttribute('download', 'molecule.json');
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        document.body.removeChild(downloadAnchor);
    });
    
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            console.log('Analyze button clicked');
            if (moleculesList.length === 0) {
                alert('No molecules generated. Please generate molecules first.');
                return;
            }
            window.location.href = '/analysis';
        });
    }
    
    closeModal.addEventListener('click', function() {
        simulationModal.style.display = 'none';
        simulationDisplay.innerHTML = '';
    });
    
    window.addEventListener('click', function(event) {
        if (event.target == simulationModal) {
            simulationModal.style.display = 'none';
            simulationDisplay.innerHTML = '';
        }
    });
    
    function createSliderValueDisplay(slider) {
        const valueDisplay = document.getElementById(`${slider.id}-value`);
        if (valueDisplay) {
            valueDisplay.textContent = slider.value;
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        }
    }
    
    createSliderValueDisplay(temperatureSlider);
    createSliderValueDisplay(uniquenessSlider);
    createSliderValueDisplay(complexitySlider);
    
    updatePagination();
});