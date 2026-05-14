<!DOCTYPE html>
<html lang="ko">

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />

  <title>Login</title>

  <link rel="stylesheet" href="../css/login.css">
</head>

<body>

  <div class="login-container">

    <div class="login-box">

      <h1>Login</h1>

      <p class="login-subtitle">
        Enter your username and password
        <br>
        to login
      </p>

      <input
        type="text"
        id="login-id"
        class="login-input"
      >

      <div class="login-link">
        Forgot Username?
      </div>

      <input
        type="password"
        id="login-password"
        class="login-input"
      >

      <div class="login-link">
        Forgot Password?
      </div>

      <button
        class="login-btn"
        onclick="login()"
      >
        Login
      </button>

    </div>

  </div>

  <script src="../js/login.js"></script>

</body>
</html>