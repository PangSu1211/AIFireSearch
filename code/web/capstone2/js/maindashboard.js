"use strict";

/* ===================== DATA ===================== */
/* 구현 쉽게 1~3층까지만 사용 */
const FLOORS_DATA = [
  {
    floor: 1,
    cameras: [
      { id: "1F-01", people: 12, confidence: 38 },
      { id: "1F-02", people: 24, confidence: 72 },
      { id: "1F-03", people: 10, confidence: 91 },
    ],
  },
  {
    floor: 2,
    cameras: [
      { id: "2F-01", people: 5, confidence: 62 },
      { id: "2F-02", people: 2, confidence: 85 },
      { id: "2F-03", people: 16, confidence: 93 },
    ],
  },
  {
    floor: 3,
    cameras: [
      { id: "3F-01", people: 24, confidence: 88 },
      { id: "3F-02", people: 0, confidence: 90 },
      { id: "3F-03", people: 13, confidence: 82 },
    ],
  },
];

let allCameras = [];
let currentPopupCamId = null;

/* ===================== HELPERS ===================== */
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

function totalPeople() {
  return FLOORS_DATA.reduce((sum, floor) => {
    return sum + floor.cameras.reduce((s, cam) => s + cam.people, 0);
  }, 0);
}

function getFloorStatus(cameras) {
  const statuses = cameras.map(cam => getStatus(cam.confidence));

  if (statuses.includes("red")) return "red";
  if (statuses.includes("yellow")) return "yellow";
  return "green";
}

function getWorstCamera(cameras) {
  return [...cameras].sort((a, b) => a.confidence - b.confidence)[0];
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

  select.innerHTML = '<option value="all">전체 층</option>';

  FLOORS_DATA.forEach(floor => {
    const option = document.createElement("option");
    option.value = floor.floor;
    option.textContent = `${floor.floor}층`;
    select.appendChild(option);
  });
}

function filterFloor(value) {
  buildGrid(value);
}

/* ===================== GRID ===================== */
function buildAllCameraList() {
  allCameras = [];

  FLOORS_DATA.forEach(floor => {
    floor.cameras.forEach(cam => {
      allCameras.push({
        ...cam,
        floor: floor.floor,
        status: getStatus(cam.confidence),
      });
    });
  });
}

function buildGrid(filterFloor = "all") {
  const grid = document.getElementById("camera-grid");
  grid.innerHTML = "";

  buildAllCameraList();

  const floorsToShow =
    filterFloor === "all"
      ? FLOORS_DATA
      : FLOORS_DATA.filter(floor => String(floor.floor) === String(filterFloor));

  grid.style.gridTemplateColumns =
    floorsToShow.length === 1
      ? "minmax(0, 1fr)"
      : `repeat(${floorsToShow.length}, minmax(0, 1fr))`;

  floorsToShow.forEach(floor => {
    const floorPeople =
      floor.cameras.reduce((sum, cam) => sum + cam.people, 0);

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
        <div class="cam-floor-tag">${floor.floor}층 · 대표 CCTV</div>
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

  document.getElementById("total-count").textContent = `${totalPeople()}명`;
}

/* ===================== NOTIFICATIONS ===================== */
function buildNotifications() {
  const alerts = [];

  FLOORS_DATA.forEach(floor => {
    floor.cameras.forEach(cam => {
      const status = getStatus(cam.confidence);

      if (status !== "green") {
        alerts.push({
          ...cam,
          floor: floor.floor,
          status,
        });
      }
    });
  });

  alerts.sort((a, b) => {
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

  if (!bell.contains(event.target) && !list.contains(event.target)) {
    closeNotifList();
  }
});

/* ===================== NOTIFICATION POPUP ===================== */
function openNotifPopup(camId) {
  currentPopupCamId = camId;

  const cam = allCameras.find(c => c.id === camId);
  if (!cam) return;

  const status = cam.status || getStatus(cam.confidence);

  const body = document.getElementById("notif-popup-body");
  const goBtn = document.getElementById("notif-go-btn");

  goBtn.style.display = "inline-flex";

  const previewHtml =
    status === "red"
      ? `
        <div class="popup-camera-preview">
          <div class="popup-camera-tag">${cam.id} · ${cam.floor}층 CCTV</div>
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

          <div class="popup-camera-confidence">${cam.confidence}%</div>
          <span class="popup-live-badge">● LIVE</span>
        </div>
      `
      : "";

  const message =
    status === "red"
      ? `카메라 신뢰도가 <b style="color:#ff4d4d">${cam.confidence}%</b>로 매우 낮습니다. 즉시 점검이 필요합니다.`
      : `카메라 신뢰도가 <b style="color:#f5c518">${cam.confidence}%</b>로 주의 수준입니다. 확인이 필요합니다.`;

  body.innerHTML = `
    ${previewHtml}

    <div class="popup-detail">
      <strong>${cam.id} — ${cam.floor}층 카메라</strong>
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
        ${statusLabel(status)} · 인원 ${cam.people}명
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

  const cam = allCameras.find(c => c.id === currentPopupCamId);
  if (!cam) return;

  window.location.href = `floordashboard.html?floor=${cam.floor}&cam=${cam.id}`;
}

/* ===================== INIT ===================== */
function init() {
  updateClock();
  setInterval(updateClock, 1000);

  buildFloorSelect();
  buildGrid("all");
  buildNotifications();
}

document.addEventListener("DOMContentLoaded", init);
