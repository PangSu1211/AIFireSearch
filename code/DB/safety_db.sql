CREATE DATABASE IF NOT EXISTS safety_system
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE safety_system;

-- 사용자
CREATE TABLE IF NOT EXISTS user (
  user_id INT NOT NULL AUTO_INCREMENT,
  username VARCHAR(50) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL,
  role ENUM('admin','operator') NOT NULL DEFAULT 'operator',
  PRIMARY KEY (user_id)
);

-- 건물
CREATE TABLE IF NOT EXISTS building (
  building_id INT NOT NULL AUTO_INCREMENT,
  building_name VARCHAR(100) NOT NULL,
  floor_count INT NOT NULL DEFAULT 1,
  PRIMARY KEY (building_id)
);

-- 층
CREATE TABLE IF NOT EXISTS floor (
  floor_id INT NOT NULL AUTO_INCREMENT,
  building_id INT NOT NULL,
  floor_num INT NOT NULL,
  floor_image VARCHAR(500),
  PRIMARY KEY (floor_id),
  UNIQUE KEY uq_floor (building_id, floor_num),
  FOREIGN KEY (building_id) REFERENCES building(building_id) ON DELETE CASCADE
);

-- 구역
CREATE TABLE IF NOT EXISTS zone (
  zone_id INT NOT NULL AUTO_INCREMENT,
  floor_id INT NOT NULL,
  zone_name VARCHAR(50) NOT NULL,
  svg_cx FLOAT,
  svg_cy FLOAT,
  svg_w FLOAT,
  svg_h FLOAT,
  PRIMARY KEY (zone_id),
  FOREIGN KEY (floor_id) REFERENCES floor(floor_id) ON DELETE CASCADE
);

-- 센서
CREATE TABLE IF NOT EXISTS sensor (
  sensor_id INT NOT NULL AUTO_INCREMENT,
  zone_id INT NOT NULL,
  sensor_type VARCHAR(50) NOT NULL DEFAULT 'mmWave',
  serial_no VARCHAR(100),
  status ENUM('active','inactive','error') NOT NULL DEFAULT 'active',
  installed_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (sensor_id),
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE CASCADE
);

-- 탐지 데이터
CREATE TABLE IF NOT EXISTS detection_data (
  data_id BIGINT NOT NULL AUTO_INCREMENT,
  zone_id INT NOT NULL,
  sensor_id INT,
  person_count INT NOT NULL DEFAULT 0,
  confidence FLOAT NOT NULL,
  source ENUM('camera','sensor','fusion') NOT NULL DEFAULT 'camera',
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (data_id),
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE CASCADE,
  FOREIGN KEY (sensor_id) REFERENCES sensor(sensor_id) ON DELETE SET NULL,
  INDEX idx_detected_at (detected_at),
  INDEX idx_zone_detected (zone_id, detected_at)
);

-- 탐지된 사람
CREATE TABLE IF NOT EXISTS detected_person (
  person_id BIGINT NOT NULL AUTO_INCREMENT,
  data_id BIGINT NOT NULL,
  x_coord FLOAT NOT NULL,
  y_coord FLOAT NOT NULL,
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (person_id),
  FOREIGN KEY (data_id) REFERENCES detection_data(data_id) ON DELETE CASCADE,
  INDEX idx_person_time (detected_at)
);

-- 화재 이벤트
CREATE TABLE IF NOT EXISTS fire_event (
  fire_id INT NOT NULL AUTO_INCREMENT,
  building_id INT NOT NULL,
  zone_id INT,
  detected_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  resolved_at DATETIME,
  status ENUM('detected','responding','resolved') NOT NULL DEFAULT 'detected',
  confidence FLOAT NOT NULL,
  snapshot_path VARCHAR(500),
  PRIMARY KEY (fire_id),
  FOREIGN KEY (building_id) REFERENCES building(building_id) ON DELETE CASCADE,
  FOREIGN KEY (zone_id) REFERENCES zone(zone_id) ON DELETE SET NULL,
  INDEX idx_fire_time (detected_at)
);