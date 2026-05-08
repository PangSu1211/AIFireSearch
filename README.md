# AIFireSearch

이후 작업 추가 예정

## docs

- 피그마 주소 : https://www.figma.com/design/hlDMKicaKes1jks8DGJKcS/Capston2?node-id=0-1&t=9m4UtF0M3w49Ypgf-0
- requirements (Google Sheet) , 관련 서비스 및 기술  : https://docs.google.com/spreadsheets/d/1uYpF3duvYlRQkQQaSm8fvD1F9VOX2kp7LKn74pYy3gY/edit?gid=0#gid=0
- requirements.pdf  : 요구사항 정의서
- erd.pdf  : ERD 설계
- usecase.pdf  : 유스케이스 다이어그램
- usecasd_spec.pdf  : 유스케이스 명세서
- system_design.pdf  : 시스템 설계도
- system_flow.png  : 시스템 흐름도

## code
### web
대충 틀만 잡고 있는 중
/*
  ===================== 테스트용 건물 구조 =====================
  여기는 화면 확인용 기본 테스트 값입니다.

  - floorCount: 전체 층 수
  - cameraCounts: 각 층에 등록할 카메라 개수

  예)
  floorCount: 3
  cameraCounts: {
    1: 10,  // 1층 카메라 10대
    2: 5,   // 2층 카메라 5대
    3: 3,   // 3층 카메라 3대
  }

  나중에 실제 구현할 때는 사용자가 입력한 층 수/카메라 수를
  localStorage 또는 서버 DB에 저장하면 이 기본값 대신 그 값이 적용됩니다.
*/
const DEFAULT_BUILDING_CONFIG = {
  floorCount: 3,
  cameraCounts: {
    1: 10, // 테스트값: 1층 카메라 10대
    2: 5,  // 테스트값: 2층 카메라 5대
    3: 3,  // 테스트값: 3층 카메라 3대
  },
};

