document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/model_metrics');
        const data = await response.json();
        
        if (data.success) {
            updateMetricsTable(data.metrics);
            updateComparisonChart(data.metrics);
        } else {
            alert(`Error loading metrics: ${data.error}`);
        }
    } catch (error) {
        alert(`Error loading metrics: ${error.message}`);
    }
    
    function updateMetricsTable(metrics) {
        const tbody = document.getElementById('metrics-table');
        tbody.innerHTML = '';
        
        for (const [model, values] of Object.entries(metrics)) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${model.toUpperCase()}</td>
                <td>${values.validity}</td>
                <td>${values.uniqueness}</td>
                <td>${values.novelty}</td>
                <td>${values.generation_time}</td>
            `;
            tbody.appendChild(row);
        }
    }
    
    function updateComparisonChart(metrics) {
        const ctx = document.getElementById('property-comparison-chart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['LogP', 'Molecular Weight', 'QED'],
                datasets: [
                    {
                        label: 'QM9',
                        data: [
                            metrics.moflow.qm9_mean_logp,
                            metrics.moflow.qm9_mean_mw,
                            metrics.moflow.qm9_mean_qed
                        ],
                        backgroundColor: 'rgba(75, 192, 192, 0.5)'
                    },
                    {
                        label: 'MoFlow',
                        data: [
                            metrics.moflow.model_mean_logp,
                            metrics.moflow.model_mean_mw,
                            metrics.moflow.model_mean_qed
                        ],
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    },
                    {
                        label: 'VAE',
                        data: [
                            metrics.vae.model_mean_logp,
                            metrics.vae.model_mean_mw,
                            metrics.vae.model_mean_qed
                        ],
                        backgroundColor: 'rgba(54, 162, 235, 0.5)'
                    }
                ]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Value' }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Property Comparison with QM9'
                    }
                }
            }
        });
    }
});