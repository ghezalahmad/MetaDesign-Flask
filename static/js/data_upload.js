document.addEventListener('DOMContentLoaded', function() {
    console.log("data_upload.js loaded");
    const form = document.querySelector('form');
    const statusMessage = document.createElement('div');
    statusMessage.className = 'mt-3';
    form.insertAdjacentElement('afterend', statusMessage);

    form.addEventListener('submit', function(event) {
        event.preventDefault();
        statusMessage.textContent = 'Uploading...';
        statusMessage.className = 'mt-3 alert alert-info';

        const formData = new FormData(form);
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusMessage.textContent = data.message;
                statusMessage.className = 'mt-3 alert alert-success';
                // Optionally, redirect or update the UI further
            } else {
                statusMessage.textContent = 'Error: ' + data.error;
                statusMessage.className = 'mt-3 alert alert-danger';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            statusMessage.textContent = 'An unexpected error occurred.';
            statusMessage.className = 'mt-3 alert alert-danger';
        });
    });
});
