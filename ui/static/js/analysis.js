document.addEventListener('DOMContentLoaded', function() {
    const propertySelect = document.getElementById('property-select');
    const analyzeBtn = document.getElementById('analyze-btn');
    const filteredTable = document.getElementById('filtered-table');
    const errorMessage = document.getElementById('error-message');
    const mwMinInput = document.getElementById('filter-mw-min');
    const mwMaxInput = document.getElementById('filter-mw-max');
    const logpMinInput = document.getElementById('filter-logp-min');
    const logpMaxInput = document.getElementById('filter-logp-max');
    const qedMinInput = document.getElementById('filter-qed-min');
    const qedMaxInput = document.getElementById('filter-qed-max');
    const tanimotoScore = document.getElementById('tanimoto-score');
    let molecules = [];
    let chart = null;

    // Debug element existence
    if (!analyzeBtn) console.error('Analyze button not found. Expected ID: analyze-btn');
    if (!errorMessage) console.error('Error message element not found. Expected ID: error-message');
    if (!filteredTable) console.error('Filtered table not found. Expected ID: filtered-table');

    // Load molecules from localStorage
    try {
        const stored = localStorage.getItem('molecules');
        if (stored) {
            molecules = JSON.parse(stored);
            molecules = molecules.map(mol => ({
                ...mol,
                molWeight: parseFloat(mol.molWeight) || null,
                logP: parseFloat(mol.logP) || null,
                h_donors: parseFloat(mol.h_donors) || null,
                h_acceptors: parseFloat(mol.h_acceptors) || null,
                rot_bonds: parseFloat(mol.rot_bonds) || null,
                qed: parseFloat(mol.qed) || null,
                sa_score: parseFloat(mol.sa_score) || null,
                validity: parseFloat(mol.validity) || null
            }));
        }
    } catch (e) {
        console.error('Error parsing molecules:', e);
        if (errorMessage) errorMessage.textContent = 'Error loading molecules.';
        return;
    }

    // Initial table render
    updateTable(molecules);

    // Initial distributions
    if (molecules.length > 0) {
        fetchDistributions(molecules);
    } else if (errorMessage) {
        errorMessage.textContent = 'No molecules available to analyze. Please generate molecules first.';
    }

    // Analyze Molecules button
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', function() {
            console.log('Analyze button clicked. Molecules:', molecules.length);
            if (molecules.length === 0) {
                if (errorMessage) errorMessage.textContent = 'No molecules generated. Please generate molecules first.';
                return;
            }

            // Get filter values
            const filters = {
                molWeight: {
                    min: parseFloat(mwMinInput.value) || -Infinity,
                    max: parseFloat(mwMaxInput.value) || Infinity
                },
                logP: {
                    min: parseFloat(logpMinInput.value) || -Infinity,
                    max: parseFloat(logpMaxInput.value) || Infinity
                },
                qed: {
                    min: parseFloat(qedMinInput.value) || -Infinity,
                    max: parseFloat(qedMaxInput.value) || Infinity
                }
            };

            // Apply filters
            const filtered = molecules.filter(mol => {
                return (
                    (mol.molWeight === null || (mol.molWeight >= filters.molWeight.min && mol.molWeight <= filters.molWeight.max)) &&
                    (mol.logP === null || (mol.logP >= filters.logP.min && mol.logP <= filters.logP.max)) &&
                    (mol.qed === null || (mol.qed >= filters.qed.min && mol.qed <= filters.qed.max))
                );
            });

            if (filtered.length === 0) {
                if (errorMessage) errorMessage.textContent = 'No molecules match the filter criteria.';
                updateTable([]);
                return;
            }

            if (errorMessage) errorMessage.textContent = '';
            updateTable(filtered);
            fetchDistributions(filtered);
        });
    }

    function updateTable(molecules) {
        if (!filteredTable) return;
        filteredTable.innerHTML = '';
        molecules.forEach(mol => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><img src="data:image/png;base64,${mol.image}" alt="Molecule" width="60"></td>
                <td>${mol.formula || '-'}</td>
                <td>${mol.molWeight != null ? mol.molWeight.toFixed(2) : '-'}</td>
                <td>${mol.logP != null ? mol.logP.toFixed(2) : '-'}</td>
                <td>${mol.qed != null ? mol.qed.toFixed(2) : '-'}</td>
                <td>${mol.sa_score != null ? mol.sa_score.toFixed(2) : '-'}</td>
            `;
            filteredTable.appendChild(row);
        });
    }

    function fetchDistributions(molecules) {
        fetch('/analyze_molecules', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ molecules })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                renderChart(data.distributions);
                if (tanimotoScore) tanimotoScore.textContent = data.diversity_score != null ? data.diversity_score.toFixed(2) : '-';
                if (errorMessage) errorMessage.textContent = '';
            } else {
                if (errorMessage) errorMessage.textContent = `Analysis error: ${data.error}`;
            }
        })
        .catch(error => {
            console.error('Error fetching distributions:', error);
            if (errorMessage) errorMessage.textContent = 'Error fetching analysis data.';
        });
    }

    function renderChart(distributions) {
        const ctx = document.getElementById('property-chart')?.getContext('2d');
        if (!ctx) {
            console.error('Chart canvas not found. Expected ID: property-chart');
            return;
        }
        if (chart) chart.destroy();

        const property = propertySelect.value;
        const values = distributions[property]?.values || [];
        if (!values.length) {
            if (errorMessage) errorMessage.textContent = 'No data available for selected property.';
            return;
        }

        const bins = 20;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const step = (max - min) / bins;
        const histogram = Array(bins).fill(0);
        values.forEach(val => {
            if (val != null) {
                const index = Math.min(Math.floor((val - min) / step), bins - 1);
                histogram[index]++;
            }
        });

        const properties = [
            { value: 'molWeight', label: 'Molecular Weight' },
            { value: 'logP', label: 'LogP' },
            { value: 'h_donors', label: 'H Donors' },
            { value: 'h_acceptors', label: 'H Acceptors' },
            { value: 'rot_bonds', label: 'Rotatable Bonds' },
            { value: 'qed', label: 'QED' },
            { value: 'sa_score', label: 'SA Score' },
            { value: 'validity', label: 'Validity' }
        ];

        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({ length: bins }, (_, i) => (min + i * step).toFixed(2)),
                datasets: [{
                    label: properties.find(p => p.value === property)?.label || property,
                    data: histogram,
                    backgroundColor: 'rgba(49, 130, 206, 0.6)',
                    borderColor: 'rgba(49, 130, 206, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { title: { display: true, text: 'Value' } },
                    y: { title: { display: true, text: 'Count' } }
                }
            }
        });
    }

    if (propertySelect) {
        propertySelect.addEventListener('change', () => {
            if (molecules.length > 0) fetchDistributions(molecules);
        });
    }
});