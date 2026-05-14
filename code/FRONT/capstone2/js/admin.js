"use strict";

/* ============================================================
   admin.js — Smart Fire Guard 관리자 설정
   localStorage 기반. 나중에 API 전환 시 read*/write* 함수만 교체.
============================================================ */

/* ===== STORAGE KEYS ===== */
const K = {
  buildingConfig:    "adminBuildingConfig",
  floorImages:       "adminFloorImages",
  zones:             "adminZones",
  cameras:           "adminCameras",
  sensors:           "adminSensors",
  thresholds:        "adminAlertThresholds",
  users:             "adminUsers",
  aiResults:         "aiCameraResults",
  cameraLayout:      "cameraLayout",
  buildingConfigLeg: "buildingConfig",
};

/* ===== READ / WRITE (교체 포인트) ===== */
function readStorage(key, fallback) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
  catch { return fallback; }
}
function writeStorage(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch { return false; }
}

/* ===== STATE ===== */
let state = {
  building: { buildingName: "", floorCount: 3, floors: [] },
  floorImages: [],
  zones: [],
  cameras: [],
  sensors: [],
  thresholds: { greenMin: 80, yellowMin: 50 },
  users: [],
  editingZone: null,
  editingCamera: null,
  editingSensor: null,
  selectedDemoCamId: null,
};

/* ===== DEFAULT CAMERAS (도면 기준) ===== */
const DEFAULT_CAMERAS = [
  { id:"1F-01", floor:1, zoneId:"", place:"엘리베이터",  cx:128, cy:150, w:160, h:100, cctvUrl:"", enabled:true, people:1,  confidence:93 },
  { id:"1F-02", floor:1, zoneId:"", place:"보안실",       cx:110, cy:520, w:140, h:95,  cctvUrl:"", enabled:true, people:2,  confidence:90 },
  { id:"1F-03", floor:1, zoneId:"", place:"안내데스크",   cx:505, cy:140, w:190, h:100, cctvUrl:"", enabled:true, people:3,  confidence:87 },
  { id:"1F-04", floor:1, zoneId:"", place:"로비 중앙",    cx:450, cy:400, w:180, h:100, cctvUrl:"", enabled:true, people:0,  confidence:64 },
  { id:"1F-05", floor:1, zoneId:"", place:"대기공간",     cx:840, cy:460, w:160, h:100, cctvUrl:"", enabled:true, people:3,  confidence:26 },
  { id:"1F-06", floor:1, zoneId:"", place:"출입구",       cx:500, cy:640, w:180, h:90,  cctvUrl:"", enabled:true, people:2,  confidence:92 },
  { id:"2F-01", floor:2, zoneId:"", place:"구내식당 좌",  cx:235, cy:460, w:170, h:105, cctvUrl:"", enabled:true, people:8,  confidence:88 },
  { id:"2F-02", floor:2, zoneId:"", place:"주방",         cx:225, cy:115, w:170, h:100, cctvUrl:"", enabled:true, people:5,  confidence:57 },
  { id:"2F-03", floor:2, zoneId:"", place:"구내식당 우",  cx:420, cy:600, w:160, h:100, cctvUrl:"", enabled:true, people:3,  confidence:91 },
  { id:"2F-04", floor:2, zoneId:"", place:"계단/엘리베이터", cx:600, cy:240, w:150, h:100, cctvUrl:"", enabled:true, people:2, confidence:58 },
  { id:"2F-05", floor:2, zoneId:"", place:"구내카페 상",  cx:830, cy:260, w:165, h:100, cctvUrl:"", enabled:true, people:4,  confidence:76 },
  { id:"2F-06", floor:2, zoneId:"", place:"구내카페 하",  cx:830, cy:530, w:165, h:100, cctvUrl:"", enabled:true, people:1,  confidence:63 },
  { id:"3F-01", floor:3, zoneId:"", place:"사무실 A",     cx:222, cy:155, w:175, h:105, cctvUrl:"", enabled:true, people:6,  confidence:95 },
  { id:"3F-02", floor:3, zoneId:"", place:"회의실",       cx:562, cy:155, w:160, h:105, cctvUrl:"", enabled:true, people:4,  confidence:81 },
  { id:"3F-03", floor:3, zoneId:"", place:"탕비실",       cx:730, cy:155, w:130, h:100, cctvUrl:"", enabled:true, people:2,  confidence:58 },
  { id:"3F-04", floor:3, zoneId:"", place:"임원실",       cx:120, cy:510, w:155, h:105, cctvUrl:"", enabled:true, people:3,  confidence:42 },
  { id:"3F-05", floor:3, zoneId:"", place:"사무실 B",     cx:440, cy:510, w:175, h:105, cctvUrl:"", enabled:true, people:1,  confidence:89 },
  { id:"3F-06", floor:3, zoneId:"", place:"서버실",       cx:730, cy:548, w:130, h:100, cctvUrl:"", enabled:true, people:5,  confidence:72 },
  { id:"3F-07", floor:3, zoneId:"", place:"복도",         cx:390, cy:365, w:175, h:65,  cctvUrl:"", enabled:true, people:0,  confidence:93 },
];

/* ===== STATUS HELPER ===== */
function getStatus(conf) {
  const t = state.thresholds;
  if (conf >= t.greenMin) return "green";
  if (conf >= t.yellowMin) return "yellow";
  return "red";
}

/* ===== LOG ===== */
function addLog(msg, ok = true) {
  const list = document.getElementById("logList");
  if (!list) return;
  const now = new Date();
  const ts = `${String(now.getHours()).padStart(2,"0")}:${String(now.getMinutes()).padStart(2,"0")}:${String(now.getSeconds()).padStart(2,"0")}`;
  const item = document.createElement("div");
  item.className = `log-item ${ok ? "log-ok" : "log-err"}`;
  item.innerHTML = `<span class="log-time">${ts}</span><span class="log-mark">${ok?"✓":"✗"}</span> ${msg}`;
  if (list.children.length === 1 && list.children[0].style.color) list.innerHTML = "";
  list.prepend(item);
  if (list.children.length > 50) list.removeChild(list.lastChild);
}

/* ===== CLOCK ===== */
function updateClock() {
  const now = new Date();
  const d = document.getElementById("sidebar-date");
  const t = document.getElementById("sidebar-time");
  if (d) d.textContent = `${now.getFullYear()}년 ${now.getMonth()+1}월 ${now.getDate()}일`;
  if (t) t.textContent = now.toLocaleTimeString("ko-KR",{hour:"2-digit",minute:"2-digit"});
}

/* ===== SUMMARY ===== */
function updateSummary() {
  const alerts = state.cameras.filter(c => getStatus(c.confidence) !== "green").length;
  document.getElementById("s-building").textContent = state.building.buildingName || "—";
  document.getElementById("s-floors").textContent   = state.building.floorCount || 0;
  document.getElementById("s-cameras").textContent  = state.cameras.length;
  document.getElementById("s-zones").textContent    = state.zones.length;
  document.getElementById("s-sensors").textContent  = state.sensors.length;
  document.getElementById("s-alerts").textContent   = alerts;
}

/* ================================================================
   1. 건물 기본 설정
================================================================ */
function applyFloorCount() {
  const n = parseInt(document.getElementById("inp-floor-count").value) || 0;
  if (n < 1 || n > 30) { alert("1~30층 사이로 입력하세요."); return; }
  const wrap = document.getElementById("floorNameRows");
  wrap.innerHTML = "";
  const existing = state.building.floors || [];
  for (let i = 1; i <= n; i++) {
    const prev = existing.find(f => f.floor === i);
    const row = document.createElement("div");
    row.className = "floor-name-row";
    row.innerHTML = `
      <span class="floor-badge">${i}층</span>
      <input type="text" data-floor="${i}" value="${prev?.name||""}" placeholder="${i}층 이름 (예: 로비)"/>`;
    wrap.appendChild(row);
  }
}

function saveBuildingConfig() {
  const name = document.getElementById("inp-building-name").value.trim();
  const cnt  = parseInt(document.getElementById("inp-floor-count").value) || 0;
  if (!name)   { alert("건물명을 입력하세요."); return; }
  if (cnt < 1) { alert("층 수를 입력하세요."); return; }

  const floors = [];
  document.querySelectorAll("#floorNameRows input[data-floor]").forEach(inp => {
    floors.push({ floor: Number(inp.dataset.floor), name: inp.value.trim() });
  });

  state.building = { buildingName: name, floorCount: cnt, floors };
  writeStorage(K.buildingConfig, state.building);

  /* 기존 maindashboard 호환 buildingConfig 도 업데이트 */
  const cameraCounts = {};
  for (let i = 1; i <= cnt; i++) {
    const fl = state.cameras.filter(c => c.floor === i);
    cameraCounts[i] = fl.length || 0;
  }
  writeStorage(K.buildingConfigLeg, { floorCount: cnt, cameraCounts });

  updateFloorSelects();
  updateSummary();
  addLog(`건물 기본 설정 저장 — ${name} (${cnt}층)`);
}

function resetBuildingConfig() {
  if (!confirm("건물 기본 설정을 초기화하시겠습니까?")) return;
  document.getElementById("inp-building-name").value = "";
  document.getElementById("inp-floor-count").value  = "";
  document.getElementById("floorNameRows").innerHTML = "";
  state.building = { buildingName:"", floorCount:3, floors:[] };
  writeStorage(K.buildingConfig, state.building);
  updateSummary();
  addLog("건물 기본 설정 초기화");
}

function loadBuildingConfig() {
  const saved = readStorage(K.buildingConfig, null);
  if (!saved) return;
  state.building = saved;
  document.getElementById("inp-building-name").value = saved.buildingName || "";
  document.getElementById("inp-floor-count").value   = saved.floorCount   || "";
  if (saved.floorCount > 0) {
    applyFloorCount();
    (saved.floors || []).forEach(f => {
      const inp = document.querySelector(`#floorNameRows input[data-floor="${f.floor}"]`);
      if (inp) inp.value = f.name;
    });
  }
}

/* ================================================================
   2. 도면 관리
================================================================ */
function onFloorImgSelect() {
  const floor = parseInt(document.getElementById("floor-img-select").value);
  const rec   = state.floorImages.find(r => r.floor === floor);
  document.getElementById("inp-floor-img").value = rec?.image || "";
  renderImgPreview(rec?.image || "");
}

function previewFloorImage() {
  renderImgPreview(document.getElementById("inp-floor-img").value.trim());
}

function renderImgPreview(src) {
  const wrap = document.getElementById("imgPreviewWrap");
  if (!src) {
    wrap.innerHTML = `<span class="img-preview-empty">경로를 입력하면 미리보기가 표시됩니다</span>`;
    return;
  }
  wrap.innerHTML = `<img src="${src}" alt="도면 미리보기" onerror="this.parentElement.innerHTML='<span class=\\'img-preview-empty\\'>이미지를 불러올 수 없습니다</span>'"/>`;
}

function saveFloorImage() {
  const floor = parseInt(document.getElementById("floor-img-select").value);
  const img   = document.getElementById("inp-floor-img").value.trim();
  if (!img) { alert("도면 경로를 입력하세요."); return; }
  const idx = state.floorImages.findIndex(r => r.floor === floor);
  if (idx >= 0) state.floorImages[idx].image = img;
  else state.floorImages.push({ floor, image: img });
  writeStorage(K.floorImages, state.floorImages);
  addLog(`도면 저장 — ${floor}층: ${img}`);
}

function deleteFloorImage() {
  const floor = parseInt(document.getElementById("floor-img-select").value);
  state.floorImages = state.floorImages.filter(r => r.floor !== floor);
  writeStorage(K.floorImages, state.floorImages);
  document.getElementById("inp-floor-img").value = "";
  renderImgPreview("");
  addLog(`도면 삭제 — ${floor}층`);
}

/* ================================================================
   3. 구역 설정
================================================================ */
function renderZoneTable() {
  const selFloor = parseInt(document.getElementById("zone-floor-select").value) || 0;
  const tbody    = document.getElementById("zoneTbody");
  tbody.innerHTML = "";
  const zones = selFloor ? state.zones.filter(z => z.floor === selFloor) : state.zones;
  zones.forEach(z => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span style="font-family:var(--font-mono)">${z.zoneId}</span></td>
      <td>${z.floor}층</td><td>${z.zoneName}</td>
      <td>${z.svgCx}</td><td>${z.svgCy}</td><td>${z.svgW}</td><td>${z.svgH}</td>
      <td style="white-space:nowrap">
        <button class="btn btn-gray btn-sm" onclick="editZone('${z.zoneId}')">수정</button>
        <button class="btn btn-danger btn-sm" style="margin-left:4px" onclick="deleteZone('${z.zoneId}')">삭제</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function addOrUpdateZone() {
  const id   = document.getElementById("inp-zone-id").value.trim();
  const name = document.getElementById("inp-zone-name").value.trim();
  const fl   = parseInt(document.getElementById("zone-floor-select").value) || 1;
  const cx   = Number(document.getElementById("inp-zone-cx").value);
  const cy   = Number(document.getElementById("inp-zone-cy").value);
  const w    = Number(document.getElementById("inp-zone-w").value);
  const h    = Number(document.getElementById("inp-zone-h").value);
  if (!id || !name) { alert("구역 ID와 구역명을 입력하세요."); return; }
  const rec = { zoneId:id, floor:fl, zoneName:name, svgCx:cx, svgCy:cy, svgW:w, svgH:h };
  const idx = state.zones.findIndex(z => z.zoneId === id);
  if (idx >= 0) state.zones[idx] = rec;
  else state.zones.push(rec);
  renderZoneTable();
  clearZoneForm();
}

function editZone(id) {
  const z = state.zones.find(z => z.zoneId === id);
  if (!z) return;
  document.getElementById("inp-zone-id").value   = z.zoneId;
  document.getElementById("inp-zone-name").value  = z.zoneName;
  document.getElementById("zone-floor-select").value = z.floor;
  document.getElementById("inp-zone-cx").value   = z.svgCx;
  document.getElementById("inp-zone-cy").value   = z.svgCy;
  document.getElementById("inp-zone-w").value    = z.svgW;
  document.getElementById("inp-zone-h").value    = z.svgH;
}

function deleteZone(id) {
  state.zones = state.zones.filter(z => z.zoneId !== id);
  renderZoneTable();
}

function clearZoneForm() {
  ["inp-zone-id","inp-zone-name","inp-zone-cx","inp-zone-cy","inp-zone-w","inp-zone-h"].forEach(id => {
    document.getElementById(id).value = "";
  });
}

function saveZones() {
  writeStorage(K.zones, state.zones);
  updateSummary();
  addLog(`구역 설정 저장 — 총 ${state.zones.length}개`);
}

/* ================================================================
   4. 카메라 관리
================================================================ */
function renderCameraTable() {
  const tbody = document.getElementById("cameraTbody");
  tbody.innerHTML = "";
  state.cameras.forEach(c => {
    const st = getStatus(c.confidence);
    const stLabel = st==="red"?"위험":st==="yellow"?"주의":"정상";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span style="font-family:var(--font-mono)">${c.id}</span></td>
      <td>${c.floor}층</td>
      <td>${c.zoneId||"—"}</td>
      <td>${c.place}</td>
      <td>${c.people}</td>
      <td>${c.confidence}%</td>
      <td><span class="tbl-badge ${st}">${stLabel}</span></td>
      <td><span class="tbl-badge ${c.enabled?"blue":"gray"}">${c.enabled?"사용":"미사용"}</span></td>
      <td style="white-space:nowrap">
        <button class="btn btn-gray btn-sm" onclick="editCamera('${c.id}')">수정</button>
        <button class="btn btn-danger btn-sm" style="margin-left:4px" onclick="deleteCamera('${c.id}')">삭제</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function addOrUpdateCamera() {
  const id      = document.getElementById("inp-cam-id").value.trim();
  const floor   = parseInt(document.getElementById("inp-cam-floor").value) || 1;
  const zoneId  = document.getElementById("inp-cam-zone").value.trim();
  const place   = document.getElementById("inp-cam-place").value.trim();
  const cctvUrl = document.getElementById("inp-cam-url").value.trim();
  const cx      = Number(document.getElementById("inp-cam-cx").value);
  const cy      = Number(document.getElementById("inp-cam-cy").value);
  const w       = Number(document.getElementById("inp-cam-w").value);
  const h       = Number(document.getElementById("inp-cam-h").value);
  const enabled = document.getElementById("inp-cam-enabled").value === "true";
  const people  = parseInt(document.getElementById("inp-cam-people").value) || 0;
  const conf    = parseInt(document.getElementById("inp-cam-conf").value);
  const confidence = isNaN(conf) ? 100 : Math.max(0, Math.min(100, conf));
  if (!id) { alert("카메라 ID를 입력하세요."); return; }
  const rec = { id, floor, zoneId, place, cx, cy, w, h, cctvUrl, enabled, people, confidence };
  const idx = state.cameras.findIndex(c => c.id === id);
  if (idx >= 0) state.cameras[idx] = rec;
  else state.cameras.push(rec);
  renderCameraTable();
  renderDemoCamGrid();
  clearCameraForm();
}

function editCamera(id) {
  const c = state.cameras.find(c => c.id === id);
  if (!c) return;
  document.getElementById("inp-cam-id").value      = c.id;
  document.getElementById("inp-cam-floor").value   = c.floor;
  document.getElementById("inp-cam-zone").value    = c.zoneId || "";
  document.getElementById("inp-cam-place").value   = c.place;
  document.getElementById("inp-cam-url").value     = c.cctvUrl || "";
  document.getElementById("inp-cam-cx").value      = c.cx;
  document.getElementById("inp-cam-cy").value      = c.cy;
  document.getElementById("inp-cam-w").value       = c.w;
  document.getElementById("inp-cam-h").value       = c.h;
  document.getElementById("inp-cam-enabled").value = String(c.enabled !== false);
  document.getElementById("inp-cam-people").value  = c.people;
  document.getElementById("inp-cam-conf").value    = c.confidence;
}

function deleteCamera(id) {
  state.cameras = state.cameras.filter(c => c.id !== id);
  renderCameraTable();
  renderDemoCamGrid();
}

function clearCameraForm() {
  ["inp-cam-id","inp-cam-floor","inp-cam-zone","inp-cam-place","inp-cam-url",
   "inp-cam-cx","inp-cam-cy","inp-cam-w","inp-cam-h","inp-cam-people","inp-cam-conf"].forEach(id => {
    document.getElementById(id).value = "";
  });
  document.getElementById("inp-cam-enabled").value = "true";
}

function saveCameras() {
  writeStorage(K.cameras, state.cameras);

  /* aiCameraResults 동기화 */
  const aiResults = state.cameras.map(c => ({
    id: c.id, floor: c.floor, people: c.people, confidence: c.confidence,
  }));
  writeStorage(K.aiResults, aiResults);

  /* cameraLayout 동기화 */
  const floorNums = [...new Set(state.cameras.map(c => c.floor))].sort((a,b)=>a-b);
  const layout = floorNums.map(fl => ({
    floor: fl,
    cameras: state.cameras.filter(c => c.floor === fl).map(c => ({ id:c.id, floor:fl })),
  }));
  writeStorage(K.cameraLayout, layout);

  /* buildingConfig 동기화 */
  const cameraCounts = {};
  floorNums.forEach(fl => { cameraCounts[fl] = state.cameras.filter(c=>c.floor===fl).length; });
  const bCfg = readStorage(K.buildingConfigLeg, {});
  writeStorage(K.buildingConfigLeg, { ...bCfg, cameraCounts });

  updateSummary();
  addLog(`카메라 저장 — 총 ${state.cameras.length}대, AI 결과 동기화 완료`);
}

function restoreDefaultCameras() {
  if (!confirm("카메라 목록을 기본값으로 복구하시겠습니까?")) return;
  state.cameras = JSON.parse(JSON.stringify(DEFAULT_CAMERAS));
  renderCameraTable();
  renderDemoCamGrid();
  addLog("카메라 기본값 복구");
}

/* ================================================================
   5. 센서 관리
================================================================ */
function renderSensorTable() {
  const tbody = document.getElementById("sensorTbody");
  tbody.innerHTML = "";
  const statusLabel = { normal:"정상", check:"점검필요", disconnected:"미연결" };
  const statusClass = { normal:"green", check:"yellow", disconnected:"gray" };
  state.sensors.forEach(s => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span style="font-family:var(--font-mono)">${s.sensorId}</span></td>
      <td>${s.floor}층</td><td>${s.zoneId||"—"}</td><td>${s.place}</td>
      <td>${s.x}</td><td>${s.y}</td>
      <td><span class="tbl-badge ${statusClass[s.status]||"gray"}">${statusLabel[s.status]||s.status}</span></td>
      <td><span class="tbl-badge ${s.enabled?"blue":"gray"}">${s.enabled?"사용":"미사용"}</span></td>
      <td style="white-space:nowrap">
        <button class="btn btn-gray btn-sm" onclick="editSensor('${s.sensorId}')">수정</button>
        <button class="btn btn-danger btn-sm" style="margin-left:4px" onclick="deleteSensor('${s.sensorId}')">삭제</button>
      </td>`;
    tbody.appendChild(tr);
  });
}

function addOrUpdateSensor() {
  const id      = document.getElementById("inp-sensor-id").value.trim();
  const floor   = parseInt(document.getElementById("inp-sensor-floor").value) || 1;
  const zoneId  = document.getElementById("inp-sensor-zone").value.trim();
  const place   = document.getElementById("inp-sensor-place").value.trim();
  const x       = Number(document.getElementById("inp-sensor-x").value);
  const y       = Number(document.getElementById("inp-sensor-y").value);
  const status  = document.getElementById("inp-sensor-status").value;
  const enabled = document.getElementById("inp-sensor-enabled").value === "true";
  if (!id) { alert("센서 ID를 입력하세요."); return; }
  const rec = { sensorId:id, floor, zoneId, place, x, y, status, enabled };
  const idx = state.sensors.findIndex(s => s.sensorId === id);
  if (idx >= 0) state.sensors[idx] = rec;
  else state.sensors.push(rec);
  renderSensorTable();
  clearSensorForm();
}

function editSensor(id) {
  const s = state.sensors.find(s => s.sensorId === id);
  if (!s) return;
  document.getElementById("inp-sensor-id").value      = s.sensorId;
  document.getElementById("inp-sensor-floor").value   = s.floor;
  document.getElementById("inp-sensor-zone").value    = s.zoneId||"";
  document.getElementById("inp-sensor-place").value   = s.place;
  document.getElementById("inp-sensor-x").value       = s.x;
  document.getElementById("inp-sensor-y").value       = s.y;
  document.getElementById("inp-sensor-status").value  = s.status;
  document.getElementById("inp-sensor-enabled").value = String(s.enabled!==false);
}

function deleteSensor(id) {
  state.sensors = state.sensors.filter(s => s.sensorId !== id);
  renderSensorTable();
}

function clearSensorForm() {
  ["inp-sensor-id","inp-sensor-floor","inp-sensor-zone","inp-sensor-place","inp-sensor-x","inp-sensor-y"].forEach(id => {
    document.getElementById(id).value = "";
  });
  document.getElementById("inp-sensor-status").value  = "normal";
  document.getElementById("inp-sensor-enabled").value = "true";
}

function saveSensors() {
  writeStorage(K.sensors, state.sensors);
  updateSummary();
  addLog(`센서 저장 — 총 ${state.sensors.length}개`);
}

/* ================================================================
   6. 알림 기준
================================================================ */
function saveThresholds() {
  const gMin = parseInt(document.getElementById("inp-green-min").value);
  const yMin = parseInt(document.getElementById("inp-yellow-min").value);
  if (isNaN(gMin)||isNaN(yMin)||yMin>=gMin) {
    alert("정상 기준이 주의 기준보다 높아야 합니다.");
    return;
  }
  state.thresholds = { greenMin: gMin, yellowMin: yMin };
  writeStorage(K.thresholds, state.thresholds);
  renderCameraTable();
  updateSummary();
  addLog(`알림 기준 저장 — 정상 ${gMin}% 이상, 주의 ${yMin}% 이상`);
}

function resetThresholds() {
  state.thresholds = { greenMin:80, yellowMin:50 };
  document.getElementById("inp-green-min").value  = 80;
  document.getElementById("inp-yellow-min").value = 50;
  writeStorage(K.thresholds, state.thresholds);
  addLog("알림 기준 기본값 복원");
}

function loadThresholds() {
  const t = readStorage(K.thresholds, { greenMin:80, yellowMin:50 });
  state.thresholds = t;
  document.getElementById("inp-green-min").value  = t.greenMin;
  document.getElementById("inp-yellow-min").value = t.yellowMin;
}

/* ================================================================
   7. 사용자 관리
================================================================ */
function renderUserTable() {
  const tbody = document.getElementById("userTbody");
  tbody.innerHTML = "";
  const roleLabel = { admin:"관리자", operator:"운영자" };
  state.users.forEach((u, idx) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span style="font-family:var(--font-mono)">${u.username}</span></td>
      <td><span class="tbl-badge ${u.role==="admin"?"blue":"gray"}">${roleLabel[u.role]||u.role}</span></td>
      <td>${u.memo||""}</td>
      <td><button class="btn btn-danger btn-sm" onclick="deleteUser(${idx})">삭제</button></td>`;
    tbody.appendChild(tr);
  });
}

function addUser() {
  const id   = document.getElementById("inp-user-id").value.trim();
  const role = document.getElementById("inp-user-role").value;
  const memo = document.getElementById("inp-user-memo").value.trim();
  if (!id) { alert("아이디를 입력하세요."); return; }
  if (state.users.find(u => u.username === id)) { alert("이미 존재하는 아이디입니다."); return; }
  state.users.push({ username:id, role, memo });
  renderUserTable();
  document.getElementById("inp-user-id").value   = "";
  document.getElementById("inp-user-memo").value = "";
}

function deleteUser(idx) {
  state.users.splice(idx, 1);
  renderUserTable();
}

function saveUsers() {
  writeStorage(K.users, state.users);
  addLog(`사용자 목록 저장 — 총 ${state.users.length}명`);
}

/* ================================================================
   8. 시연 데이터 설정
================================================================ */
function renderDemoCamGrid() {
  const grid = document.getElementById("demoCamGrid");
  if (!grid) return;
  grid.innerHTML = "";
  state.cameras.forEach(c => {
    const st = getStatus(c.confidence);
    const dotColor = st==="red"?"var(--status-red)":st==="yellow"?"var(--status-yellow)":"var(--status-green)";
    const chip = document.createElement("div");
    chip.className = `demo-cam-chip${state.selectedDemoCamId===c.id?" selected":""}`;
    chip.dataset.id = c.id;
    chip.innerHTML = `<span class="chip-dot" style="background:${dotColor}"></span>${c.id}`;
    chip.addEventListener("click", () => {
      state.selectedDemoCamId = c.id;
      document.getElementById("demo-conf").value   = c.confidence;
      document.getElementById("demo-people").value = c.people;
      renderDemoCamGrid();
    });
    grid.appendChild(chip);
  });
}

function setDemoStatus(type) {
  const confMap = { danger:26, warning:58, normal:90 };
  document.getElementById("demo-conf").value = confMap[type] ?? 90;
}

function applyDemoChange() {
  const id      = state.selectedDemoCamId;
  const conf    = parseInt(document.getElementById("demo-conf").value);
  const people  = parseInt(document.getElementById("demo-people").value) || 0;
  if (!id) { alert("카메라를 선택하세요."); return; }
  if (isNaN(conf)) { alert("신뢰도를 입력하세요."); return; }
  const cam = state.cameras.find(c => c.id === id);
  if (!cam) return;
  cam.confidence = Math.max(0, Math.min(100, conf));
  cam.people = people;
  renderCameraTable();
  renderDemoCamGrid();
  saveCameras();
  addLog(`시연 데이터 — ${id}: 신뢰도 ${cam.confidence}%, 인원 ${people}명`);
}

function resetAllDemo() {
  if (!confirm("모든 카메라를 기본값으로 초기화하시겠습니까?")) return;
  state.cameras = JSON.parse(JSON.stringify(DEFAULT_CAMERAS));
  renderCameraTable();
  renderDemoCamGrid();
  saveCameras();
  addLog("시연 데이터 전체 초기화");
}

/* ================================================================
   공통: 층 선택 드롭다운 업데이트
================================================================ */
function updateFloorSelects() {
  const cnt = state.building.floorCount || 3;
  const selectors = ["floor-img-select","zone-floor-select"];
  selectors.forEach(selId => {
    const sel = document.getElementById(selId);
    if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = "";
    for (let i = 1; i <= cnt; i++) {
      const opt = document.createElement("option");
      opt.value = i; opt.textContent = `${i}층`;
      sel.appendChild(opt);
    }
    if (cur && parseInt(cur) <= cnt) sel.value = cur;
  });
}

/* ================================================================
   INIT
================================================================ */
function init() {
  updateClock();
  setInterval(updateClock, 1000);

  /* 데이터 로드 */
  state.thresholds = readStorage(K.thresholds, { greenMin:80, yellowMin:50 });
  loadThresholds();

  loadBuildingConfig();
  updateFloorSelects();

  state.floorImages = readStorage(K.floorImages, []);
  state.zones       = readStorage(K.zones, []);
  state.cameras     = readStorage(K.cameras, null);
  if (!state.cameras || state.cameras.length === 0) {
    state.cameras = JSON.parse(JSON.stringify(DEFAULT_CAMERAS));
  }
  state.sensors = readStorage(K.sensors, []);
  state.users   = readStorage(K.users, [
    { username:"admin", role:"admin", memo:"관리자" },
    { username:"operator1", role:"operator", memo:"관제 담당자" },
  ]);

  renderZoneTable();
  renderCameraTable();
  renderSensorTable();
  renderUserTable();
  renderDemoCamGrid();
  updateSummary();
}

window.addEventListener("DOMContentLoaded", init);

/* 전역 노출 (HTML onclick 용) */
window.applyFloorCount    = applyFloorCount;
window.saveBuildingConfig = saveBuildingConfig;
window.resetBuildingConfig= resetBuildingConfig;
window.onFloorImgSelect   = onFloorImgSelect;
window.previewFloorImage  = previewFloorImage;
window.saveFloorImage     = saveFloorImage;
window.deleteFloorImage   = deleteFloorImage;
window.renderZoneTable    = renderZoneTable;
window.addOrUpdateZone    = addOrUpdateZone;
window.editZone           = editZone;
window.deleteZone         = deleteZone;
window.clearZoneForm      = clearZoneForm;
window.saveZones          = saveZones;
window.addOrUpdateCamera  = addOrUpdateCamera;
window.editCamera         = editCamera;
window.deleteCamera       = deleteCamera;
window.clearCameraForm    = clearCameraForm;
window.saveCameras        = saveCameras;
window.restoreDefaultCameras = restoreDefaultCameras;
window.addOrUpdateSensor  = addOrUpdateSensor;
window.editSensor         = editSensor;
window.deleteSensor       = deleteSensor;
window.clearSensorForm    = clearSensorForm;
window.saveSensors        = saveSensors;
window.saveThresholds     = saveThresholds;
window.resetThresholds    = resetThresholds;
window.addUser            = addUser;
window.deleteUser         = deleteUser;
window.saveUsers          = saveUsers;
window.setDemoStatus      = setDemoStatus;
window.applyDemoChange    = applyDemoChange;
window.resetAllDemo       = resetAllDemo;
