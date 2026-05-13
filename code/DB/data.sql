USE safety_system;
 
 INSERT INTO user (username, password, role) VALUES
  ('admin', '$2b$12$placeholder_hash_admin', 'admin'),
  ('operator1', '$2b$12$placeholder_hash_op1', 'operator');

INSERT INTO building (building_name, floor_count) VALUES
  ('청출어람관', 3);

INSERT INTO floor (building_id, floor_num, floor_image) VALUES
  (1, 1, '/image/1층도면.png'),
  (1, 2, '/image/2층도면.png'),
  (1, 3, '/image/3층도면.png');

INSERT INTO zone (floor_id, zone_name, svg_cx, svg_cy, svg_w, svg_h) VALUES
  (1, '1F-01', 100, 100, 80, 80),
  (1, '1F-02', 200, 100, 80, 80),
  (1, '1F-03', 300, 100, 80, 80),
  (1, '1F-04', 400, 100, 80, 80),
  (1, '1F-05', 500, 100, 80, 80),
  (1, '1F-06', 600, 100, 80, 80),
  (2, '2F-01', 100, 100, 80, 80),
  (2, '2F-02', 200, 100, 80, 80),
  (2, '2F-03', 300, 100, 80, 80),
  (2, '2F-04', 400, 100, 80, 80),
  (2, '2F-05', 500, 100, 80, 80),
  (2, '2F-06', 600, 100, 80, 80),
  (3, '3F-01', 100, 100, 80, 80),
  (3, '3F-02', 200, 100, 80, 80),
  (3, '3F-03', 300, 100, 80, 80),
  (3, '3F-04', 400, 100, 80, 80),
  (3, '3F-05', 500, 100, 80, 80),
  (3, '3F-06', 600, 100, 80, 80),
  (3, '3F-07', 700, 100, 80, 80);

INSERT INTO sensor (zone_id, sensor_type, serial_no) VALUES
  (1,  'mmWave', 'SN-1F-01'),
  (2,  'mmWave', 'SN-1F-02'),
  (3,  'mmWave', 'SN-1F-03'),
  (7,  'mmWave', 'SN-2F-01'),
  (8,  'mmWave', 'SN-2F-02'),
  (13, 'mmWave', 'SN-3F-01');
  
-- detection_data
INSERT INTO detection_data (zone_id, sensor_id, person_count, confidence, source, detected_at) VALUES
  (1,  1,    1,  93.0, 'fusion',  NOW() - INTERVAL 30 SECOND),
  (2,  2,    2,  90.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (3,  3,    3,  87.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (4,  NULL, 0,  64.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (5,  NULL, 3,  26.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (6,  NULL, 2,  92.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (7,  4,    8,  88.0, 'fusion',  NOW() - INTERVAL 30 SECOND),
  (8,  5,    5,  57.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (9,  NULL, 3,  91.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (10, NULL, 2,  58.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (11, NULL, 4,  76.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (12, NULL, 1,  63.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (13, 6,    6,  95.0, 'fusion',  NOW() - INTERVAL 30 SECOND),
  (14, NULL, 4,  81.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (15, NULL, 2,  58.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (16, NULL, 3,  42.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (17, NULL, 1,  89.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (18, NULL, 5,  72.0, 'camera',  NOW() - INTERVAL 30 SECOND),
  (19, NULL, 0,  93.0, 'camera',  NOW() - INTERVAL 30 SECOND);
 
-- detected_person
INSERT INTO detected_person (data_id, x_coord, y_coord) VALUES
  (1,  120.0, 130.0),
  (2,  210.0, 110.0),
  (2,  230.0, 140.0),
  (3,  305.0, 105.0),
  (3,  330.0, 125.0),
  (3,  310.0, 150.0),
  (5,  510.0, 108.0),
  (5,  535.0, 130.0),
  (5,  520.0, 155.0),
  (6,  615.0, 115.0),
  (6,  640.0, 140.0),
  (7,  108.0, 108.0),
  (7,  130.0, 120.0),
  (7,  115.0, 140.0),
  (7,  145.0, 108.0),
  (7,  160.0, 130.0),
  (7,  108.0, 160.0),
  (7,  140.0, 158.0),
  (7,  165.0, 155.0),
  (8,  208.0, 110.0),
  (8,  225.0, 128.0),
  (8,  210.0, 148.0),
  (8,  240.0, 115.0),
  (8,  235.0, 145.0),
  (13, 110.0, 110.0),
  (13, 135.0, 125.0),
  (13, 115.0, 148.0),
  (13, 155.0, 112.0),
  (13, 160.0, 140.0),
  (13, 140.0, 155.0),
  (16, 408.0, 108.0),
  (16, 430.0, 130.0),
  (16, 415.0, 155.0);
 
-- fire_event
INSERT INTO fire_event (building_id, zone_id, detected_at, resolved_at, status, confidence, snapshot_path) VALUES
  (1, 5,  NOW() - INTERVAL 10 MINUTE, NULL,                      'responding', 82.5, '/snapshots/fire_1F05.jpg'),
  (1, 16, NOW() - INTERVAL 2 HOUR,    NOW() - INTERVAL 1 HOUR,   'resolved',   91.0, '/snapshots/fire_3F04.jpg');
 
-- 과거 이력 (시간대별 차트용)
INSERT INTO detection_data (zone_id, person_count, confidence, source, detected_at) VALUES
  (1,  2, 91.0, 'camera', NOW() - INTERVAL 1 HOUR),
  (7,  6, 85.0, 'fusion', NOW() - INTERVAL 1 HOUR),
  (13, 4, 93.0, 'fusion', NOW() - INTERVAL 1 HOUR),
  (1,  3, 88.0, 'camera', NOW() - INTERVAL 2 HOUR),
  (7,  9, 79.0, 'fusion', NOW() - INTERVAL 2 HOUR),
  (13, 5, 90.0, 'fusion', NOW() - INTERVAL 2 HOUR),
  (1,  1, 94.0, 'camera', NOW() - INTERVAL 3 HOUR),
  (7,  4, 88.0, 'fusion', NOW() - INTERVAL 3 HOUR),
  (13, 2, 95.0, 'fusion', NOW() - INTERVAL 3 HOUR);