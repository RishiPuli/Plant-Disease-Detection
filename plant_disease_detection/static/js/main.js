document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const preview = document.getElementById('preview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultSection = document.querySelector('.result-section');
    const resultImage = document.getElementById('resultImage');
    const prediction = document.getElementById('prediction');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const errorMessage = document.getElementById('errorMessage');
    const recommendationText = document.getElementById('recommendationText');
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const closeModal = document.querySelector('.close-modal');
    const distributionChart = document.getElementById('distributionChart');
    const trendChart = document.getElementById('trendChart');

    const recommendations = {
        'Healthy': [
            'Continue regular watering schedule',
            'Maintain good air circulation',
            'Regular pruning of dead leaves',
            'Monitor for any changes'
        ],
        'Powdery': [
            'Remove affected leaves',
            'Improve air circulation',
            'Apply fungicide if severe',
            'Avoid overhead watering',
            'Keep leaves dry'
        ],
        'Rust': [
            'Remove and destroy infected leaves',
            'Apply appropriate fungicide',
            'Improve plant spacing',
            'Avoid wetting leaves',
            'Monitor other plants for signs'
        ]
    };

    async function updateAnalytics() {
        try {
            const response = await fetch('/update_analytics');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            
            if (data.pie_chart) {
                const pieChartData = JSON.parse(data.pie_chart);
                Plotly.newPlot('distributionChart', pieChartData.data, pieChartData.layout);
            }
            
            if (data.trend_chart) {
                const trendChartData = JSON.parse(data.trend_chart);
                Plotly.newPlot('trendChart', trendChartData.data, trendChartData.layout);
            }
        } catch (error) {
            console.error('Error updating analytics:', error);
        }
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 5000);
    }

    // Handle drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-color)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--secondary-color)';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--secondary-color)';
        handleFile(e.dataTransfer.files[0]);
    });

    // Handle click upload
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file) {
            showError('Please select a file');
            return;
        }

        const validTypes = ['image/jpeg', 'image/jpg'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a JPEG image file');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            preview.style.width = '150px';
            preview.style.height = 'auto';
            analyzeBtn.disabled = false;
            errorMessage.style.display = 'none';
        };
        reader.onerror = () => {
            showError('Error reading file');
        };
        reader.readAsDataURL(file);
    }

    // Handle image modal
    resultImage.addEventListener('click', () => {
        modal.style.display = 'flex';
        modalImg.src = resultImage.src;
    });

    closeModal.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });

    // Handle analysis
    analyzeBtn.addEventListener('click', async () => {
        try {
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            errorMessage.style.display = 'none';

            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: preview.src
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            // Update UI with results
            resultImage.src = data.image_path;
            resultImage.style.width = '100%';
            resultImage.style.height = 'auto';
            prediction.textContent = data.prediction;
            confidenceBar.style.width = `${data.confidence}%`;
            confidenceValue.textContent = `${data.confidence.toFixed(1)}%`;

            // Add recommendations
            const recs = recommendations[data.prediction] || [];
            recommendationText.innerHTML = recs.map(rec => `<p>• ${rec}</p>`).join('');

            // Show results
            resultSection.style.display = 'block';

            // Update analytics
            await updateAnalytics();

            // Scroll to results
            resultSection.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            showError('Error analyzing image: ' + error.message);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Plant';
        }
    });

    // Initial analytics update
    updateAnalytics();
}); 