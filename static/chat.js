document.addEventListener('DOMContentLoaded', function() {
    // DOM element references
    const messageForm = document.getElementById('messageForm');
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const modelType = document.getElementById('modelType');
    const decodeMethod = document.getElementById('decodeMethod');
    const temperature = document.getElementById('temperature');
    const tempValue = document.getElementById('tempValue');
    const topP = document.getElementById('topP');
    const topPValue = document.getElementById('topPValue');
    const beamSize = document.getElementById('beamSize');
    const beamValue = document.getElementById('beamValue');
    const settingsToggle = document.getElementById('settingsToggle');
    const settingsPanel = document.getElementById('settingsPanel');
    
    // Parameter containers
    const tempContainer = document.getElementById('tempContainer');
    const topPContainer = document.getElementById('topPContainer');
    const beamContainer = document.getElementById('beamContainer');
    
    // Accordion elements
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    
    // Initialize first accordion section as open
    const firstAccordionContent = document.querySelector('.accordion-content');
    const firstToggleIcon = document.querySelector('.toggle-icon');
    if (firstAccordionContent && firstToggleIcon) {
        firstAccordionContent.classList.add('open');
        firstToggleIcon.classList.add('open');
    }

    // Dark mode functionality
    const darkModeToggle = document.getElementById('darkModeToggle');
    
    // Function to set theme based on preference
    function setTheme(isDark) {
        if (isDark) {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        } else {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
    }
    
    // Initialize theme from stored preference
    function initializeTheme() {
        // Check for saved theme preference or use system preference
        const savedTheme = localStorage.getItem('theme');
        
        if (savedTheme === 'dark') {
            setTheme(true);
        } else if (savedTheme === 'light') {
            setTheme(false);
        } else {
            // If no saved preference, use system preference
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            setTheme(prefersDark);
        }
    }
    
    // Toggle dark mode
    darkModeToggle.addEventListener('click', function() {
        const isDark = document.documentElement.classList.contains('dark');
        setTheme(!isDark);
    });
    
    // Initialize theme
    initializeTheme();
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches);
        }
    });
    
    // Session Management
    let sessionId = null;
    
    function initializeSession() {
        fetch('/api/start-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        })
        .then(response => response.json())
        .then(data => {
            sessionId = data.session_id;
            console.log("Session initialized:", sessionId);
        })
        .catch(error => {
            console.error('Error initializing session:', error);
        });
    }
    
    // Helper Functions
    
    // Calculate a natural typing delay based on message length
    function calculateTypingDelay(message) {
        const charactersPerSecond = 10;
        const messageLength = message.length;
        const thinkingTime = 500;
        const typingTime = (messageLength / charactersPerSecond) * 1000;
        const maxTime = 1000;
        
        return Math.min(Math.max(thinkingTime + typingTime, 1500), maxTime);
    }
    
    // Add message to chat
    function addMessage(text, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'flex mb-4 ' + (isUser ? 'justify-end' : '');
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = isUser 
            ? 'bg-blue-500 text-white rounded-lg p-3 max-w-xs md:max-w-md'
            : 'bg-gray-200 rounded-lg p-3 max-w-xs md:max-w-md';
        
        bubbleDiv.textContent = text;
        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Add typing indicator
    function addTypingIndicator() {
        const indicatorDiv = document.createElement('div');
        indicatorDiv.className = 'flex mb-4 typing-indicator-container';
        indicatorDiv.id = 'typingIndicator';
        
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'bg-gray-200 rounded-lg p-3';
        
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        
        bubbleDiv.appendChild(indicator);
        indicatorDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(indicatorDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    // Input validation function
    function validateInput(message, settings) {
        const errors = [];
        
        // Message validation
        if (!message || message.trim() === '') {
            errors.push("Message cannot be empty");
            return errors;
        }
        
        if (message.length > 500) {
            errors.push("Message is too long (maximum 500 characters)");
        }
        
        // Settings validation
        if (settings.decode_method === 'top-p') {
            const temp = parseFloat(settings.temperature);
            if (isNaN(temp) || temp < 0.1 || temp > 2) {
                errors.push("Temperature must be between 0.1 and 2");
            }
            
            const topP = parseFloat(settings.top_p);
            if (isNaN(topP) || topP < 0.1 || topP > 1) {
                errors.push("Top-p must be between 0.1 and 1");
            }
        } else if (settings.decode_method === 'beam') {
            const beamSize = parseInt(settings.beam_size);
            if (isNaN(beamSize) || beamSize < 2 || beamSize > 10) {
                errors.push("Beam size must be between 2 and 10");
            }
        }
        
        return errors;
    }
    
    // Sanitize input to prevent XSS
    function sanitizeInput(input) {
        return input
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/&/g, "&amp;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#x27;");
    }
    
    // Settings panel toggle
    settingsToggle.addEventListener('click', function() {
        settingsPanel.classList.toggle('open');
    });
    
    // Update value displays
    temperature.addEventListener('input', function() {
        tempValue.textContent = this.value;
    });
    
    topP.addEventListener('input', function() {
        topPValue.textContent = this.value;
    });
    
    beamSize.addEventListener('input', function() {
        beamValue.textContent = this.value;
    });
    
    // Show/hide parameter containers based on decode method
    decodeMethod.addEventListener('change', function() {
        tempContainer.classList.add('hidden');
        topPContainer.classList.add('hidden');
        beamContainer.classList.add('hidden');
        
        if (this.value === 'top-p') {
            tempContainer.classList.remove('hidden');
            topPContainer.classList.remove('hidden');
        } else if (this.value === 'beam') {
            beamContainer.classList.remove('hidden');
        }
    });
    
    // Accordion functionality
    accordionHeaders.forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling;
            const icon = this.querySelector('.toggle-icon');
            
            content.classList.toggle('open');
            icon.classList.toggle('open');
        });
    });
    
    // Handle message submission
    messageForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Clear input immediately
        messageInput.value = '';
        
        // Prepare ONLY the basic settings first
        const settings = {
            message: sanitizeInput(message),
            model_type: modelType.value,
            decode_method: decodeMethod.value,
            session_id: sessionId
        };
        
        // Then add ONLY the parameters relevant to the selected decode method
        if (settings.decode_method === 'top-p') {
            // Only add temperature and top-p for top-p sampling
            settings.temperature = parseFloat(temperature.value);
            settings.top_p = parseFloat(topP.value);
        } 
        else if (settings.decode_method === 'beam') {
            // Only add beam size for beam search
            settings.beam_size = parseInt(beamSize.value);
            // Explicitly NOT including temperature or top_p
        }
        // No additional parameters for greedy search
        
        // Validate input
        const validationErrors = validateInput(message, settings);
        if (validationErrors.length > 0) {
            addMessage(`Error: ${validationErrors.join('. ')}`, false);
            return;
        }
        
        // Add user message to chat
        addMessage(message, true);
        
        // Show typing indicator
        addTypingIndicator();
        
        let apiResponse = null;
        let apiError = null;
        
        // Start API request (we'll handle the timing separately)
        const fetchPromise = fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: sanitizeInput(message),
                ...settings
            })
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Something went wrong');
                });
            }
            return response.json();
        })
        .then(data => {
            apiResponse = data;
            if (data.session_id) {
                sessionId = data.session_id;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            apiError = error.message;
        });
        
        try {
            // While the API is processing, we'll calculate how long the typing indicator should show
            const minTypingTime = 1500; // Minimum typing time in milliseconds
            
            // Start timing
            const startTime = new Date().getTime();
            
            // Wait for API response
            await fetchPromise;
            
            // Calculate elapsed time
            const elapsedTime = new Date().getTime() - startTime;
            
            // Calculate natural typing time based on response length
            let typingTime = minTypingTime;
            if (apiResponse && apiResponse.response) {
                typingTime = calculateTypingDelay(apiResponse.response);
            }
            
            // If API was faster than our typing time, wait the remaining time
            const remainingTime = typingTime - elapsedTime;
            if (remainingTime > 0) {
                await new Promise(resolve => setTimeout(resolve, remainingTime));
            }
            
            // Remove typing indicator
            removeTypingIndicator();
            
            // Display the response or error
            if (apiError) {
                addMessage(`Error: ${apiError}`, false);
            } else if (apiResponse) {
                addMessage(apiResponse.response, false);
            } else {
                addMessage('Sorry, something went wrong. Please try again.', false);
            }
        } catch (error) {
            // Remove typing indicator
            removeTypingIndicator();
            console.error('Network error:', error);
            addMessage('Network error. Please check your connection and try again.', false);
        }
    });
    
    // Initialization
    initializeSession();
    
    // Show Top-p controls by default since that's our default method
    tempContainer.classList.remove('hidden');
    topPContainer.classList.remove('hidden');
    beamContainer.classList.add('hidden');
});