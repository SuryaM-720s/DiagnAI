// Tab switching functionality
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`button[onclick="switchTab('${tabName}')"]`).classList.add('active');

    document.querySelectorAll('.auth-form').forEach(form => form.classList.remove('active'));
    document.getElementById(`${tabName}-form`).classList.add('active');
}

// Login functionality
async function handleLogin(event) {
    event.preventDefault();
    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    // Sample login logic placeholder for database interaction
    // Replace with actual API calls to validate user credentials
    console.log('Attempting login with:', { email, password });

    if (email === "test@example.com" && password === "password123") {
        alert("Login successful!");
    } else {
        alert("Login failed. Check your credentials.");
    }
}

// Signup functionality
async function handleSignup(event) {
    event.preventDefault();
    const username = document.getElementById('signup-username').value;
    const email = document.getElementById('signup-email').value;
    const password = document.getElementById('signup-password').value;

    // Sample signup logic placeholder for database interaction
    console.log('Creating new account:', { username, email, password });
    alert("Sign-up successful! You may now log in.");
}
