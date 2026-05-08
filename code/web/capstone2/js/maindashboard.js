"use strict";

/* ============================================================
   Dynamic Dashboard
   - 층 수, 카메라 개수: buildingConfig / cameraLayout 기준
   - 인원 수, 신뢰도: aiCameraResults 기준
   - 색상, 알림, 총원: JS 자동 계산
============================================================ */

const BUILDING_CONFIG_KEY = "buildingConfig";
const CAMERA_LAYOUT_KEY = "cameraLayout";
const AI_RESULTS_KEY = "aiCameraResults";

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

/*
  ===================== 테스트용 AI 결과값 =====================
  여기는 AI가 아직 연결되지 않았을 때 화면에 보여주기 위한 테스트 데이터입니다.

  실제 구현에서는 AI/서버가 아래 형태로 보내주면 됩니다.

  {
    id: "1F-01",       // 카메라 ID
    floor: 1,          // 층
    people: 12,        // AI가 판단한 인원 수
    confidence: 38     // AI가 판단한 카메라 신뢰도
  }

  화면은 이 confidence 숫자를 보고 자동으로 색을 정합니다.
  - 80 이상: 초록색
  - 50 이상 80 미만: 노란색
  - 50 미만: 빨간색

  people 값도 자동으로 총원, 층별 인원, 주의 필요 인원에 반영됩니다.
*/
const DEFAULT_AI_RESULTS = [
  // ===== 1층 테스트 카메라 10대  이런 느낌으로 ai가 값을 줄 수 있게 하면 될 듯 =====
  { id: "1F-01", floor: 1, people: 1, confidence: 93 },  // 정상
  { id: "1F-02", floor: 1, people: 2, confidence: 90 },  // 정상
  { id: "1F-03", floor: 1, people: 3, confidence: 87 },  // 정상
  { id: "1F-04", floor: 1, people: 2, confidence: 64 },  // 주의
  { id: "1F-05", floor: 1, people: 3, confidence: 90 },  // 정상
  { id: "1F-06", floor: 1, people: 0, confidence: 92 },  // 정상
  { id: "1F-07", floor: 1, people: 4, confidence: 57 },  // 주의
  { id: "1F-08", floor: 1, people: 2, confidence: 26 },  // 위험
  { id: "1F-09", floor: 1, people: 3, confidence: 52 },  // 주의
  { id: "1F-10", floor: 1, people: 0, confidence: 82 },  // 정상

  // ===== 2층 테스트 카메라 5대 =====
  { id: "2F-01", floor: 2, people: 0, confidence: 88 },  // 정상
  { id: "2F-02", floor: 2, people: 1, confidence: 76 },  // 주의
  { id: "2F-03", floor: 2, people: 3, confidence: 91 },  // 정상
  { id: "2F-04", floor: 2, people: 2, confidence: 58 },  // 주의
  { id: "2F-05", floor: 2, people: 1, confidence: 63 },  // 주의

  // ===== 3층 테스트 카메라 3대 =====
  { id: "3F-01", floor: 3, people: 1, confidence: 95 },  // 정상
  { id: "3F-02", floor: 3, people: 2, confidence: 81 },  // 정상
  { id: "3F-03", floor: 3, people: 0, confidence: 58 },  // 주의
];

let FLOORS_DATA = [];
let allCameras = [];
let currentPopupCamId = null;

/* ===================== COMMON DATA ===================== */
function safeJsonParse(value, fallback) {
  try {
    if (!value) return fallback;
    return JSON.parse(value);
  } catch (error) {
    return fallback;
  }
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function normalizeAIResult(result) {
  return {
    id: String(result.id),
    floor: Number(result.floor),
    people: Number(result.people ?? 0),
    confidence: Number(result.confidence ?? result.trust ?? 100),
  };
}

/* 실제 API 연결 시에는 이 함수 안에서 fetch('/api/ai-results') 같은 방식으로 서버 값을 받아오면 됩니다. */
function readAIResults() {
  const saved = safeJsonParse(localStorage.getItem(AI_RESULTS_KEY), null);

  if (!Array.isArray(saved)) {
    return DEFAULT_AI_RESULTS.map(normalizeAIResult);
  }

  return saved.map(normalizeAIResult);
}

function createLayoutFromConfig(config) {
  const floorCount = Number(config?.floorCount ?? DEFAULT_BUILDING_CONFIG.floorCount);
  const cameraCounts = config?.cameraCounts ?? DEFAULT_BUILDING_CONFIG.cameraCounts;

  const layout = [];

  for (let floor = 1; floor <= floorCount; floor++) {
    const cameraCount = Number(cameraCounts[floor] ?? cameraCounts[String(floor)] ?? 0);
    const cameras = [];

    for (let index = 1; index <= cameraCount; index++) {
      cameras.push({
        id: `${floor}F-${pad2(index)}`,
        floor,
      });
    }

    layout.push({
      floor,
      cameras,
    });
  }

  return layout;
}

function readCameraLayout() {
  const savedLayout = safeJsonParse(localStorage.getItem(CAMERA_LAYOUT_KEY), null);

  if (Array.isArray(savedLayout)) {
    return savedLayout.map(floor => ({
      floor: Number(floor.floor),
      cameras: Array.isArray(floor.cameras)
        ? floor.cameras.map(camera => ({
            id: String(camera.id),
            floor: Number(camera.floor ?? floor.floor),
          }))
        : [],
    }));
  }

  const savedConfig = safeJsonParse(
    localStorage.getItem(BUILDING_CONFIG_KEY),
    DEFAULT_BUILDING_CONFIG
  );

  return createLayoutFromConfig(savedConfig);
}

function buildFloorsData() {
  const layout = readCameraLayout();
  const aiResults = readAIResults();
  const resultMap = new Map(aiResults.map(result => [result.id, result]));

  FLOORS_DATA = layout.map(floor => ({
    floor: Number(floor.floor),
    cameras: floor.cameras.map(camera => {
      const result = resultMap.get(camera.id);

      return {
        id: camera.id,
        floor: Number(floor.floor),
        people: Number(result?.people ?? 0),
        confidence: Number(result?.confidence ?? 100),
      };
    }),
  }));

  allCameras = FLOORS_DATA.flatMap(floor =>
    floor.cameras.map(camera => ({
      ...camera,
      floor: floor.floor,
      status: getStatus(camera.confidence),
    }))
  );
}

/*
  테스트용:
  콘솔에서 아래처럼 실행하면 층 수/카메라 수가 바로 바뀜.
  setBuildingConfig(5, {1:3, 2:4, 3:2, 4:5, 5:1})

  AI 결과 테스트:
  setCameraAIResults([{id:"1F-01", floor:1, people:8, confidence:44}])
*/
function setBuildingConfig(floorCount, cameraCounts) {
  localStorage.setItem(
    BUILDING_CONFIG_KEY,
    JSON.stringify({ floorCount, cameraCounts })
  );

  localStorage.removeItem(CAMERA_LAYOUT_KEY);
  refreshDashboard();
}

function setCameraAIResults(results) {
  localStorage.setItem(AI_RESULTS_KEY, JSON.stringify(results));
  refreshDashboard();
}

window.setBuildingConfig = setBuildingConfig;
window.setCameraAIResults = setCameraAIResults;

function resetDashboardDefaults() {
  localStorage.removeItem(BUILDING_CONFIG_KEY);
  localStorage.removeItem(CAMERA_LAYOUT_KEY);
  localStorage.removeItem(AI_RESULTS_KEY);
  location.reload();
}

window.resetDashboardDefaults = resetDashboardDefaults;


/* ===================== STATUS ===================== */
function getStatus(confidence) {
  if (confidence >= 80) return "green";
  if (confidence >= 50) return "yellow";
  return "red";
}

function statusLabel(status) {
  if (status === "red") return "위험";
  if (status === "yellow") return "주의";
  return "정상";
}

function getFloorStatus(cameras) {
  const statuses = cameras.map(camera => getStatus(camera.confidence));

  if (statuses.includes("red")) return "red";
  if (statuses.includes("yellow")) return "yellow";
  return "green";
}

function getWorstCamera(cameras) {
  if (cameras.length === 0) {
    return {
      id: "-",
      people: 0,
      confidence: 100,
    };
  }

  return [...cameras].sort((a, b) => a.confidence - b.confidence)[0];
}

function getTotalPeople() {
  return FLOORS_DATA.reduce((sum, floor) => {
    return sum + floor.cameras.reduce((s, camera) => s + camera.people, 0);
  }, 0);
}

/* ===================== CLOCK ===================== */
function updateClock() {
  const now = new Date();

  const dateStr =
    `${now.getFullYear()}년 ${now.getMonth() + 1}월 ${now.getDate()}일`;

  const timeStr =
    now.toLocaleTimeString("ko-KR", {
      hour: "2-digit",
      minute: "2-digit",
    });

  document.getElementById("sidebar-date").textContent = dateStr;
  document.getElementById("sidebar-time").textContent = timeStr;
}

/* ===================== FLOOR SELECT ===================== */
function buildFloorSelect() {
  const select = document.getElementById("floor-select");
  const currentValue = select.value || "all";

  select.innerHTML = '<option value="all">전체 층</option>';

  FLOORS_DATA.forEach(floor => {
    const option = document.createElement("option");
    option.value = floor.floor;
    option.textContent = `${floor.floor}층`;
    select.appendChild(option);
  });

  const canKeepValue =
    currentValue === "all" ||
    FLOORS_DATA.some(floor => String(floor.floor) === String(currentValue));

  select.value = canKeepValue ? currentValue : "all";
}

function filterFloor(value) {
  buildGrid(value);
}

/* ===================== GRID ===================== */
function buildGrid(filterFloor = "all") {
  const grid = document.getElementById("camera-grid");
  const view = document.getElementById("view-building");

  grid.innerHTML = "";

  const floorsToShow =
    filterFloor === "all"
      ? FLOORS_DATA
      : FLOORS_DATA.filter(floor => String(floor.floor) === String(filterFloor));

  const columnCount =
    floorsToShow.length <= 1
      ? 1
      : Math.min(floorsToShow.length, 3);

  grid.style.gridTemplateColumns =
    `repeat(${columnCount}, minmax(0, 1fr))`;

  view.style.overflowY = floorsToShow.length > 3 ? "auto" : "hidden";

  floorsToShow.forEach(floor => {
    const floorPeople =
      floor.cameras.reduce((sum, camera) => sum + camera.people, 0);

    const floorStatus = getFloorStatus(floor.cameras);
    const worstCamera = getWorstCamera(floor.cameras);

    const card = document.createElement("div");
    card.className = `camera-card status-${floorStatus}`;
    card.dataset.floor = floor.floor;

    const alertHtml =
      floorStatus === "red"
        ? `<div class="cam-alert">신뢰도 하락<br />즉시 확인 필요</div>`
        : "";

    card.innerHTML = `
      <div class="cam-feed">
        <div class="cam-floor-tag">${floor.floor}층 CCTV</div>
        <div class="cam-status-dot"></div>

        <span class="cam-icon">
          <svg
            width="42"
            height="42"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="1.5"
          >
            <path d="M23 7l-7 5 7 5V7z" />
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
          </svg>
        </span>

        ${alertHtml}

        <div class="cam-confidence">${worstCamera.confidence}%</div>
      </div>

      <div class="cam-info">
        <span class="cam-stat">
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
            <circle cx="9" cy="7" r="4" />
          </svg>
          ${floorPeople}명
        </span>

        <span class="cam-stat">
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
          >
            <path d="M1 6l11 6 11-6M1 12l11 6 11-6" />
          </svg>
          ${floor.floor}층
        </span>
      </div>
    `;

    card.addEventListener("click", () => {
      window.location.href = `floordashboard.html?floor=${floor.floor}`;
    });

    grid.appendChild(card);
  });

  document.getElementById("total-count").textContent = `${getTotalPeople()}명`;
}

/* ===================== NOTIFICATIONS ===================== */
function buildNotifications() {
  const alerts = allCameras
    .filter(camera => camera.status !== "green")
    .sort((a, b) => {
      const order = { red: 0, yellow: 1 };
      return order[a.status] - order[b.status] || a.confidence - b.confidence;
    });

  const listEl = document.getElementById("notif-list-items");
  const emptyEl = document.getElementById("notif-list-empty");
  const bellDot = document.getElementById("bell-dot");
  const headerBadge = document.getElementById("alert-badge");

  listEl.innerHTML = "";

  if (alerts.length === 0) {
    emptyEl.classList.remove("hidden");
    bellDot.classList.remove("active");
    headerBadge.classList.add("hidden");
    return;
  }

  emptyEl.classList.add("hidden");
  bellDot.classList.add("active");
  headerBadge.classList.remove("hidden");

  alerts.forEach(alert => {
    const item = document.createElement("div");
    item.className = "notif-item";

    item.innerHTML = `
      <span class="notif-dot ${alert.status}"></span>

      <div class="notif-item-text">
        <div class="notif-item-title">${alert.id} · ${alert.floor}층</div>
        <div class="notif-item-sub">
          신뢰도 ${alert.confidence}% · 인원 ${alert.people}명 · ${statusLabel(alert.status)}
        </div>
      </div>

      <span class="notif-item-action">확인 →</span>
    `;

    item.addEventListener("click", () => {
      closeNotifList();
      openNotifPopup(alert.id);
    });

    listEl.appendChild(item);
  });
}

function toggleNotifList() {
  document.getElementById("notif-list").classList.toggle("hidden");
}

function closeNotifList() {
  document.getElementById("notif-list").classList.add("hidden");
}

document.addEventListener("click", event => {
  const bell = document.getElementById("notif-bell");
  const list = document.getElementById("notif-list");

  if (!bell || !list) return;

  if (!bell.contains(event.target) && !list.contains(event.target)) {
    closeNotifList();
  }
});

/* ===================== NOTIFICATION POPUP ===================== */
function openNotifPopup(camId) {
  currentPopupCamId = camId;

  const camera = allCameras.find(item => item.id === camId);
  if (!camera) return;

  const status = camera.status;

  const body = document.getElementById("notif-popup-body");
  const goBtn = document.getElementById("notif-go-btn");

  goBtn.style.display = "inline-flex";

  const previewHtml =
    status === "red"
      ? `
        <div class="popup-camera-preview">
          <div class="popup-camera-tag">${camera.id} · ${camera.floor}층 CCTV</div>
          <div class="popup-camera-status-dot"></div>

          <span class="popup-camera-icon">
            <svg
              width="44"
              height="44"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
            >
              <path d="M23 7l-7 5 7 5V7z" />
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
            </svg>
          </span>

          <div class="popup-camera-confidence">${camera.confidence}%</div>
          <span class="popup-live-badge">● LIVE</span>
        </div>
      `
      : "";

  const message =
    status === "red"
      ? `카메라 신뢰도가 <b style="color:#ff4d4d">${camera.confidence}%</b>로 매우 낮습니다. 즉시 점검이 필요합니다.`
      : `카메라 신뢰도가 <b style="color:#f5c518">${camera.confidence}%</b>로 주의 수준입니다. 확인이 필요합니다.`;

  body.innerHTML = `
    ${previewHtml}

    <div class="popup-detail">
      <strong>${camera.id} — ${camera.floor}층 카메라</strong>
      ${message}

      <div class="popup-status ${status}">
        <svg
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" y1="8" x2="12" y2="12" />
          <line x1="12" y1="16" x2="12.01" y2="16" />
        </svg>
        ${statusLabel(status)} · 인원 ${camera.people}명
      </div>
    </div>
  `;

  document.getElementById("notif-overlay").classList.remove("hidden");
}

function closeNotifPopup() {
  document.getElementById("notif-overlay").classList.add("hidden");
  currentPopupCamId = null;
}

function goToCamera() {
  if (!currentPopupCamId) return;

  const camera = allCameras.find(item => item.id === currentPopupCamId);
  if (!camera) return;

  window.location.href = `floordashboard.html?floor=${camera.floor}&cam=${camera.id}`;
}

/* ===================== REFRESH ===================== */
function refreshDashboard() {
  buildFloorsData();
  buildFloorSelect();

  const select = document.getElementById("floor-select");
  const selectedFloor = select?.value || "all";

  buildGrid(selectedFloor);
  buildNotifications();
}

/* ===================== INIT ===================== */
function init() {
  updateClock();
  setInterval(updateClock, 1000);

  refreshDashboard();
}

window.addEventListener("storage", event => {
  if (
    event.key === AI_RESULTS_KEY ||
    event.key === BUILDING_CONFIG_KEY ||
    event.key === CAMERA_LAYOUT_KEY
  ) {
    refreshDashboard();
  }
});

document.addEventListener("DOMContentLoaded", init);