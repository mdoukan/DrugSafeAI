// Main JavaScript file for DrugSafe AI

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the logo heart animation
    initLogoHeart();
    
    // Initialize form validation
    initFormValidation();
});

// Initialize the logo heart animation
function initLogoHeart() {
    const logoHeart = document.querySelector('.logo-heart');
    if (!logoHeart) return;
    
    // Add heart pattern inside
    const heartPattern = document.createElement('div');
    heartPattern.className = 'heartbeat-pattern';
    logoHeart.appendChild(heartPattern);
    
    // Make logo clickable to go to homepage
    logoHeart.addEventListener('click', function() {
        window.location.href = window.location.origin;
    });
    
    // Add heart pulse effect on hover
    logoHeart.addEventListener('mouseenter', function() {
        this.style.animationDuration = '0.6s';
    });
    
    logoHeart.addEventListener('mouseleave', function() {
        this.style.animationDuration = '1.5s';
    });
}

// Initialize form validation
function initFormValidation() {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    // Add "was-validated" class to form when user tries to submit
    form.addEventListener('submit', function(event) {
        if (!this.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        }
        
        this.classList.add('was-validated');
    }, false);
    
    // Highlight inputs on focus
    const formControls = document.querySelectorAll('.form-control, .form-select');
    formControls.forEach(control => {
        control.addEventListener('focus', function() {
            this.parentElement.classList.add('input-focus');
        });
        
        control.addEventListener('blur', function() {
            this.parentElement.classList.remove('input-focus');
        });
    });
    
    // Make form labels more interactive
    const formLabels = document.querySelectorAll('.form-label');
    formLabels.forEach(label => {
        const icon = document.createElement('span');
        icon.className = 'label-icon';
        
        // Add emoji icon based on label content if it doesn't already have one
        if (!label.textContent.trim().startsWith('🌸') && 
            !label.textContent.trim().startsWith('🦋') && 
            !label.textContent.trim().startsWith('🌼') && 
            !label.textContent.trim().startsWith('📋') && 
            !label.textContent.trim().startsWith('💊') && 
            !label.textContent.trim().startsWith('🎂') && 
            !label.textContent.trim().startsWith('⚖️') && 
            !label.textContent.trim().startsWith('📏') && 
            !label.textContent.trim().startsWith('👥') && 
            !label.textContent.trim().startsWith('🔬')) {
            
            // Choose an emoji based on the label content
            let emoji = '✨';
            const labelText = label.textContent.toLowerCase();
            
            if (labelText.includes('age')) emoji = '🎂';
            else if (labelText.includes('weight')) emoji = '⚖️';
            else if (labelText.includes('height')) emoji = '📏';
            else if (labelText.includes('gender') || labelText.includes('sex')) emoji = '👥';
            else if (labelText.includes('history')) emoji = '📋';
            else if (labelText.includes('medication')) emoji = '💊';
            else if (labelText.includes('lab') || labelText.includes('test')) emoji = '🔬';
            
            label.prepend(emoji + ' ');
        }
        
        // Add animation on hover
        label.addEventListener('mouseenter', function() {
            this.style.transform = 'translateX(5px)';
            this.style.color = '#ec4899';
        });
        
        label.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.color = '';
        });
    });
}

// Function to copy error message to clipboard
function copyError() {
    const errorText = document.getElementById('errorText').textContent;
    navigator.clipboard.writeText(errorText).then(() => {
        alert('Error message copied to clipboard');
    });
}

// Handle anatomy image loading error
function handleAnatomyImageError() {
    const anatomyImg = document.getElementById('anatomyImage');
    if (anatomyImg) {
        anatomyImg.src = 'https://via.placeholder.com/300x500?text=Human+Anatomy+Image+Not+Available';
    }
} 