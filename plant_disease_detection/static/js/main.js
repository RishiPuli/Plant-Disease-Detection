document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const preview = document.getElementById('preview');
    const errorMessage = document.getElementById('errorMessage');
    const uploadForm = document.getElementById('uploadForm');
    const resultSection = document.querySelector('.result-section');
    const resultImage = document.getElementById('resultImage');
    const prediction = document.getElementById('prediction');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceValue = document.getElementById('confidenceValue');
    const recommendationText = document.getElementById('recommendationText');
    const imageModal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
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

    // Make drop zone clickable
    dropZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    // Handle file input change
    fileInput.addEventListener('change', handleFileSelect, false);

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        if (!fileInput.files.length) {
            showError('Please select an image first');
            return;
        }
        handleAnalyze();
    });

    // Handle analyze button click
    analyzeBtn.addEventListener('click', handleAnalyze, false);

    // Handle image click for modal
    resultImage.addEventListener('click', openModal);
    closeModal.addEventListener('click', closeModalFunc);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.match('image.*')) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    analyzeBtn.disabled = false;
                    errorMessage.style.display = 'none';
                };
                reader.readAsDataURL(file);
            } else {
                showError('Please select a valid image file (JPEG)');
            }
        }
    }

    function handleAnalyze() {
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);
        
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                showError(data.message || 'An error occurred during analysis');
            } else {
                displayResults(data);
            }
        })
        .catch(error => {
            showError('An error occurred while processing your request: ' + error.message);
            console.error('Error:', error);
        })
        .finally(() => {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-microscope"></i> Analyze Plant';
        });
    }

    function displayResults(data) {
        resultSection.style.display = 'block';
        resultImage.src = data.image_url;
        prediction.textContent = data.class;
        prediction.className = 'diagnosis-text ' + data.class.toLowerCase();
        
        const confidence = (data.confidence * 100).toFixed(1);
        confidenceBar.style.width = confidence + '%';
        confidenceValue.textContent = confidence + '%';
        
        // Set recommendation text based on the class
        let recommendation = '';
        switch(data.class.toLowerCase()) {
            case 'healthy':
                recommendation = 'Your plant appears to be healthy! Continue with regular care and monitoring.';
                break;
            case 'powdery':
                recommendation = 'Your plant shows signs of powdery mildew. Consider:\n' +
                    '1. Improving air circulation\n' +
                    '2. Applying fungicide treatment\n' +
                    '3. Removing affected leaves\n' +
                    '4. Maintaining proper humidity levels';
                break;
            case 'rust':
                recommendation = 'Your plant shows signs of rust disease. Recommended actions:\n' +
                    '1. Remove and destroy infected leaves\n' +
                    '2. Apply appropriate fungicide\n' +
                    '3. Improve plant spacing for better air circulation\n' +
                    '4. Avoid overhead watering';
                break;
        }
        recommendationText.innerHTML = recommendation.replace(/\n/g, '<br>');
        
        // Scroll to results
        resultSection.scrollIntoView({ behavior: 'smooth' });

        // Update analytics
        updateAnalytics();
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        analyzeBtn.disabled = true;
        preview.style.display = 'none';
    }

    function openModal() {
        imageModal.style.display = 'block';
        modalImage.src = resultImage.src;
    }

    function closeModalFunc() {
        imageModal.style.display = 'none';
    }

    // Close modal when clicking outside the image
    window.onclick = function(event) {
        if (event.target == imageModal) {
            closeModalFunc();
        }
    }

    // Initial analytics update
    updateAnalytics();
}); 