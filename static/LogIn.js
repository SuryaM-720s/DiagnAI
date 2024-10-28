function handleLogin(event) {
    event.preventDefault();

    // Perform your login validation here (e.g., check email and password)
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    // For demonstration purposes, assume login is successful
    if (email && password) { // You would replace this with your actual validation logic
        localStorage.setItem('isLoggedIn', 'true'); // Set logged in status
        window.location.href = 'homepage.html'; // Redirect to homepage
    } else {
        // Handle login failure (e.g., show an error message)
        alert('Login failed. Please check your credentials.');
    }
}
