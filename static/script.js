document.addEventListener("DOMContentLoaded", function() {
    const sidebar = document.querySelector(".sidebar");
    const menuIcon = document.querySelector(".menu-icon");
    const closeBtn = document.querySelector(".close-btn");

    // Function to toggle sidebar
    menuIcon.addEventListener("click", function() {
        sidebar.classList.toggle("active");
    });

    // Function to close sidebar
    closeBtn.addEventListener("click", function() {
        sidebar.classList.remove("active");
    });

    // Close sidebar when clicking outside
    document.addEventListener("click", function(event) {
        if (!sidebar.contains(event.target) && !menuIcon.contains(event.target)) {
            sidebar.classList.remove("active");
        }
    });
});
