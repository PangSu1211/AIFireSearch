CREATE DATABASE IF NOT EXISTS safety_system
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE safety_system;

-- 사용자
-- 관리자가 직접 계정을 생성해서 부여하는 방식, 회원가입 없음
CREATE TABLE IF NOT EXISTS user (
  user_id INT NOT NULL AUTO_INCREMENT,       -- 사용자 고유 번호
  username VARCHAR(50) NOT NULL UNIQUE,      -- 아이디, 중복 불가
  password VARCHAR(255) NOT NULL,            -- 비밀번호, bcrypt 해시값 저장
  role ENUM('admin','operator') NOT NULL DEFAULT 'operator', -- 권한
  PRIMARY KEY (user_id)
);

-- 건물
CREATE TABLE IF NOT EXISTS building (
  building_id INT NOT NULL AUTO_INCREMENT,   -- 건물 고유 번호
  building_name VARCHAR(100) NOT NULL,       -- 건물 이름
  floor_count INT NOT NULL DEFAULT 1,        -- 총 층수
  PRIMARY KEY (building_id)
);

-- 층
-- 같은 건물에 같은 층 번호 중복 불가
CREATE TABLE IF NOT EXISTS floor (
  floor_id INT NOT NULL AUTO_INCREMENT,      -- 층 고유 번호
  building_id INT NOT NULL,                  -- 소속 건물 번호
  floor_num INT NOT NULL,                    -- 층 번호
  floor_image VARCHAR(500),                  -- 단면도 이미지 경로
  PRIMARY KEY (floor_id),
  UNIQUE KEY uq_floor (building_id, floor_num),
  FOREIGN KEY (building_id) REFERENCES building(building_id) ON DELETE CASCADE
);

-- 구역
-- zone_name은 JS 코드의 카메라 ID와 동일하게 맞춰야 함
CREATE TABLE IF NOT EXISTS zone (
  zone_id INT NOT NULL AUTO_INCREMENT,       -- 구역 고유 번호
  floor_id INT NOT NULL,                     -- 소속 층 번호
  zone_name VARCHAR(50) NOT NULL,            -- 구역 이름
  svg_cx FLOAT,                              -- SVG 중심 X 좌표
  svg_cy FLOAT,                              -- SVG 중심 Y 좌표
  svg_w FLOAT,                               -- 구역 박스 너비
  svg_h FLOAT,                               -- 구역 박스 높이
  PRIMARY KEY (zone_id),
  FOREIGN KEY (floor_id) REFERENCES floor(floor_id) ON DELETE CASCADE
);

-- 센서
-- 신뢰도 하락 시 카메라 대신 센서 중심으로 인원 파악
CREATE TABLE IF NOT EXISTS sensor (
  sensor_id INT NOT NULL AUTO_INCREMENT,     -- 센서 고유 번호
  zone_id INT NOT NULL,                      -- 설치된 구역 번호
  sensor_type VARCHAR(50) NOT NULL DEFAULT 'mmWave', -- 센서 종류
  serial_no VARCHAR(100),                    -- 센서 시리얼 번호
  status ENUM('active','inactive','error') NOT NULL DEFAULT 'active', -- 센서 상태
  installed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 설치 시각
  PRIMARY KEY (sensor_id),
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE CASCADE
);

-- 탐지 데이터
-- YOLO AI가 분석할 때마다 저장되는 핵심 테이블
-- source: camera / sensor / fusion
-- confidence: 80 이상 정상 / 50~79 주의 / 50 미만 위험
CREATE TABLE IF NOT EXISTS detection_data (
  data_id BIGINT NOT NULL AUTO_INCREMENT,    -- 탐지 데이터 고유 번호
  zone_id INT NOT NULL,                      -- 탐지된 구역 번호
  sensor_id INT,                             -- 센서 번호, 영상만 분석한 경우 NULL
  person_count INT NOT NULL DEFAULT 0,       -- 탐지된 인원 수
  confidence FLOAT NOT NULL,                 -- AI 신뢰도
  source ENUM('camera','sensor','fusion') NOT NULL DEFAULT 'camera', -- 데이터 출처
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 분석 시각
  PRIMARY KEY (data_id),
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE CASCADE,
  FOREIGN KEY (sensor_id) REFERENCES sensor(sensor_id) ON DELETE SET NULL,
  INDEX idx_detected_at (detected_at),
  INDEX idx_zone_detected (zone_id, detected_at)
);

-- 탐지된 사람
-- 단면도 위에 점으로 위치 표시할 때 사용
CREATE TABLE IF NOT EXISTS detected_person (
  person_id BIGINT NOT NULL AUTO_INCREMENT,  -- 인원 고유 번호
  data_id BIGINT NOT NULL,                   -- 소속 탐지 데이터 번호
  x_coord FLOAT NOT NULL,                    -- SVG 기준 X 좌표
  y_coord FLOAT NOT NULL,                    -- SVG 기준 Y 좌표
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 탐지 시각
  PRIMARY KEY (person_id),
  FOREIGN KEY (data_id) REFERENCES detection_data(data_id) ON DELETE CASCADE,
  INDEX idx_person_time (detected_at)
);

-- 화재 이벤트
-- detected → responding → resolved 단계로 상태 관리
-- resolved_at: 해결 전까지 NULL, 해결되면 UPDATE로 시각 입력
CREATE TABLE IF NOT EXISTS fire_event (
  fire_id INT NOT NULL AUTO_INCREMENT,       -- 화재 이벤트 고유 번호
  building_id INT NOT NULL,                  -- 발생 건물 번호
  zone_id INT,                               -- 발생 구역 번호, 위치 미확인 시 NULL
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, -- 감지 시각
  resolved_at DATETIME,                      -- 해결 시각, 진행 중이면 NULL
  status ENUM('detected','responding','resolved') NOT NULL DEFAULT 'detected', -- 진행 상태
  confidence FLOAT NOT NULL,                 -- 감지 당시 AI 신뢰도
  snapshot_path VARCHAR(500),                -- 감지 순간 캡처 이미지 경로
  PRIMARY KEY (fire_id),
  FOREIGN KEY (building_id) REFERENCES building(building_id) ON DELETE CASCADE,
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE SET NULL,
  INDEX idx_fire_time (detected_at)
);