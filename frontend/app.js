// Wait for DOM content to load before hooking up interactivity
document.addEventListener('DOMContentLoaded', () => {
    console.log("Quantum Ledger UI Engine Initialized.");
    
    // Future interactivity hooks can go here
    // Example: Tab switching logic for the top nav
    const tabs = document.querySelectorAll('.tabs li');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', (e) => {
            tabs.forEach(t => t.classList.remove('active'));
            e.target.classList.add('active');
        });
    });
});
