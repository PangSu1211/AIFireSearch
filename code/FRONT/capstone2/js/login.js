function login(){

  const id =
    document.getElementById('login-id').value;

  const password =
    document.getElementById('login-password').value;

  if(id === '' || password === ''){

    alert('아이디와 비밀번호를 입력해주세요.');

    return;
  }

  /*
    TODO:
    서버 로그인 API 연결
  */

  /* 로그인 성공 시 이동 */
  window.location.href =
    "maindashboard.html";
}