/* ============================================================
   app.js  –  건물 현황 CCTV 모니터링 대시보드
   Features:
     ⓐ 층 수에 따라 그리드 자동 조정 (카메라 수 → 컬럼 수)
     ⓑ 알림 기능 (벨 아이콘 + 드롭다운)
     ⓒ 공지 클릭 → 팝업
     ⓓ 팝업 확인 → 해당 카메라로 이동
============================================================ */

"use strict";

/* ===================== DATA ===================== */
// 카메라 신뢰도: 0~100, 80 이상 = green, 50~79 = yellow, 0~49 = red
const FLOORS_DATA = [
  {
    floor: 1, cameras: [
      { id: "1F-01", people: 12, confidence: 38 },  // RED
      { id: "1F-02", people: 24, confidence: 72 },  // YELLOW
      { id: "1F-03", people: 10, confidence: 91 },  // GREEN
    ]
  },
  {
    floor: 2, cameras: [
      { id: "2F-01", people: 5,  confidence: 46 },  // RED
      { id: "2F-02", people: 2,  confidence: 85 },  // GREEN
      { id: "2F-03", people: 16, confidence: 93 },  // GREEN
    ]
  },
  {
    floor: 3, cameras: [
      { id: "3F-01", people: 24, confidence: 88 },  // GREEN
      { id: "3F-02", people: 0,  confidence: 90 },  // GREEN
      { id: "3F-03", people: 13, confidence: 82 },  // GREEN
    ]
  },
  {
    floor: 4, cameras: [
      { id: "4F-01", people: 8,  confidence: 61 },  // YELLOW
      { id: "4F-02", people: 31, confidence: 55 },  // YELLOW
    ]
  },
  {
    floor: 5, cameras: [
      { id: "5F-01", people: 19, confidence: 44 },  // RED
      { id: "5F-02", people: 7,  confidence: 78 },  // YELLOW
      { id: "5F-03", people: 22, confidence: 95 },  // GREEN
    ]
  },
  {
    floor: 6, cameras: [
      { id: "6F-01", people: 14, confidence: 88 },  // GREEN
      { id: "6F-02", people: 9,  confidence: 84 },  // GREEN
    ]
  },
  {
    floor: 7, cameras: [
      { id: "7F-01", people: 3,  confidence: 42 },  // RED
      { id: "7F-02", people: 11, confidence: 67 },  // YELLOW
      { id: "7F-03", people: 18, confidence: 91 },  // GREEN
    ]
  },
  {
    floor: 8, cameras: [
      { id: "8F-01", people: 27, confidence: 73 },  // YELLOW
      { id: "8F-02", people: 5,  confidence: 88 },  // GREEN
      { id: "8F-03", people: 6,  confidence: 96 },  // GREEN
    ]
  },
  {
    floor: 9, cameras: [
      { id: "9F-01", people: 20, confidence: 33 },  // RED
      { id: "9F-02", people: 4,  confidence: 89 },  // GREEN
    ]
  },
];

/* ===================== HELPERS ===================== */
function getStatus(confidence) {
  if (confidence >= 80) return "green";
  if (confidence >= 50) return "yellow";
  return "red";
}

function statusLabel(s) {
  return s === "green" ? "정상" : s === "yellow" ? "주의" : "위험";
}

function totalPeople() {
  return FLOORS_DATA.reduce((sum, f) =>
    sum + f.cameras.reduce((s, c) => s + c.people, 0), 0);
}

/* ===================== CLOCK ===================== */
function updateClock() {
  const now = new Date();
  const dateStr = `${now.getFullYear()}년 ${now.getMonth()+1}월 ${now.getDate()}일`;
  const timeStr = now.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit" });
  document.getElementById("sidebar-date").textContent = dateStr;
  document.getElementById("sidebar-time").textContent = timeStr;
}

/* ===================== GRID COLUMN CALC ===================== */
// ⓐ: 카메라 총 개수에 따라 자동으로 열 수를 조정
function calcGridCols(totalCams) {
  if (totalCams <= 4)  return 2;
  if (totalCams <= 9)  return 3;
  if (totalCams <= 16) return 4;
  if (totalCams <= 25) return 5;
  return 6;
}

/* ===================== BUILD GRID ===================== */
let allCameras = [];   // flat list for lookup

function buildGrid(filterFloor = "all") {
  const grid = document.getElementById("camera-grid");
  grid.innerHTML = "";
  allCameras = [];

  const floorsToShow = filterFloor === "all"
    ? FLOORS_DATA
    : FLOORS_DATA.filter(f => String(f.floor) === String(filterFloor));

  let totalCams = 0;
  floorsToShow.forEach(f => { totalCams += f.cameras.length; });

  // ⓐ: 동적 컬럼 수
  const cols = calcGridCols(totalCams);
  grid.className = `camera-grid cols-${cols}`;

  floorsToShow.forEach(floor => {
    floor.cameras.forEach(cam => {
      const status = getStatus(cam.confidence);
      allCameras.push({ ...cam, floor: floor.floor, status });

      const card = document.createElement("div");
      card.className = `camera-card status-${status}`;
      card.dataset.camId = cam.id;
      card.dataset.floor = floor.floor;

      card.innerHTML = `
        <div class="cam-feed">
          <div class="cam-floor-tag">${floor.floor}F · ${cam.id}</div>
          <div class="cam-status-dot"></div>
          <span class="cam-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M23 7l-7 5 7 5V7z"/>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
          </span>
          <div class="cam-confidence">${cam.confidence}%</div>
        </div>
        <div class="cam-info">
          <span class="cam-stat">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
            ${cam.people}
          </span>
          <span class="cam-stat">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M1 6l11 6 11-6M1 12l11 6 11-6"/>
            </svg>
            ${cam.confidence}
          </span>
        </div>
      `;

      // ⓒ: 카드 클릭 → 팝업
      card.addEventListener("click", () => openNotifPopup(cam.id));
      grid.appendChild(card);
    });
  });

  // 총 인원 업데이트
  document.getElementById("total-count").textContent = totalPeople() + "명";
}

/* ===================== FLOOR FILTER ===================== */
// ⓐ: 층 선택 드롭다운 구성
function buildFloorSelect() {
  const sel = document.getElementById("floor-select");
  sel.innerHTML = '<option value="all">전체 층</option>';
  FLOORS_DATA.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f.floor;
    opt.textContent = `${f.floor}층`;
    sel.appendChild(opt);
  });
}

function filterFloor(val) {
  buildGrid(val);
}

/* ===================== NOTIFICATIONS ===================== */
// ⓑ: 알림 목록 구성 (신뢰도가 낮은 카메라)
function buildNotifications() {
  const alerts = [];
  FLOORS_DATA.forEach(f => {
    f.cameras.forEach(cam => {
      const status = getStatus(cam.confidence);
      if (status !== "green") {
        alerts.push({
          camId: cam.id,
          floor: f.floor,
          confidence: cam.confidence,
          status,
          people: cam.people,
        });
      }
    });
  });

  // Sort: red first, then yellow
  alerts.sort((a, b) => {
    const order = { red: 0, yellow: 1 };
    return order[a.status] - order[b.status];
  });

  const listEl = document.getElementById("notif-list-items");
  const emptyEl = document.getElementById("notif-list-empty");
  const bellDot  = document.getElementById("bell-dot");

  listEl.innerHTML = "";

  if (alerts.length === 0) {
    emptyEl.classList.remove("hidden");
    bellDot.classList.remove("active");
    return;
  }

  bellDot.classList.add("active");
  emptyEl.classList.add("hidden");

  alerts.forEach(a => {
    const item = document.createElement("div");
    item.className = "notif-item";
    item.innerHTML = `
      <span class="notif-dot ${a.status}"></span>
      <div class="notif-item-text">
        <div class="notif-item-title">${a.camId} · ${a.floor}층</div>
        <div class="notif-item-sub">신뢰도 ${a.confidence}% · 인원 ${a.people}명 · ${statusLabel(a.status)}</div>
      </div>
      <span class="notif-item-action">확인 →</span>
    `;
    // ⓒⓓ: 알림 항목 클릭 → 팝업 열기
    item.addEventListener("click", () => {
      closeNotifList();
      openNotifPopup(a.camId);
    });
    listEl.appendChild(item);
  });
}

/* ===================== NOTIFICATION DROPDOWN ===================== */
// ⓑ: 벨 클릭 토글
function toggleNotifList() {
  const list = document.getElementById("notif-list");
  list.classList.toggle("hidden");
}

function closeNotifList() {
  document.getElementById("notif-list").classList.add("hidden");
}

// 외부 클릭 시 닫기
document.addEventListener("click", (e) => {
  const bell = document.getElementById("notif-bell");
  const list = document.getElementById("notif-list");
  if (!bell.contains(e.target) && !list.contains(e.target)) {
    closeNotifList();
  }
});

/* ===================== POPUP ===================== */
let currentPopupCamId = null;

// ⓒ: 팝업 열기
function openNotifPopup(camId) {
  currentPopupCamId = camId;
  const cam = allCameras.find(c => c.camId === camId || c.id === camId);
  if (!cam) return;

  const status = cam.status || getStatus(cam.confidence);
  const body = document.getElementById("notif-popup-body");

  const messages = {
    red:    `카메라 신뢰도가 <b style="color:#ff4d4d">${cam.confidence}%</b>로 매우 낮습니다. 즉시 점검이 필요합니다.`,
    yellow: `카메라 신뢰도가 <b style="color:#f5c518">${cam.confidence}%</b>로 주의 수준입니다. 확인이 필요합니다.`,
    green:  `카메라 상태가 정상입니다. (신뢰도 ${cam.confidence}%)`,
  };

  body.innerHTML = `
    <strong>${cam.id || cam.camId} — ${cam.floor}층 카메라</strong>
    ${messages[status]}
    <div class="popup-status ${status === "green" ? "" : status}">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      ${statusLabel(status)} · 인원 ${cam.people}명
    </div>
  `;

  document.getElementById("notif-overlay").classList.remove("hidden");
}

// ⓒ: 팝업 닫기
function closeNotifPopup() {
  document.getElementById("notif-overlay").classList.add("hidden");
  currentPopupCamId = null;
}

// ⓓ: 팝업 → "해당 카메라로 이동" → 카메라 카드로 스크롤 + 하이라이트
function goToCamera() {
  closeNotifPopup();
  if (!currentPopupCamId) return;

  // 건물 현황 뷰로 전환
  const buildingNav = document.querySelector('[data-view="building"]');
  if (buildingNav) switchView("building", buildingNav);

  // 해당 카메라 카드 찾기
  const card = document.querySelector(`[data-cam-id="${currentPopupCamId}"]`);
  if (!card) return;

  // 부드럽게 스크롤
  setTimeout(() => {
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    // 하이라이트 애니메이션
    card.classList.add("cam-highlight");
    setTimeout(() => card.classList.remove("cam-highlight"), 2000);
  }, 120);

  currentPopupCamId = null;
}

// 하이라이트 스타일 (동적 추가)
const style = document.createElement("style");
style.textContent = `
  .cam-highlight {
    animation: cam-hl 2s ease !important;
  }
  @keyframes cam-hl {
    0%   { outline: 3px solid #fff; outline-offset: 2px; }
    60%  { outline: 3px solid rgba(255,255,255,0.6); }
    100% { outline: 3px solid transparent; }
  }
`;
document.head.appendChild(style);

/* ===================== VIEW SWITCHING ===================== */
function switchView(viewName, navEl) {
  // 네비 활성화
  document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
  if (navEl) navEl.classList.add("active");

  // 뷰 전환
  document.querySelectorAll(".view").forEach(v => {
    v.classList.add("hidden");
    v.classList.remove("active");
  });

  const target = document.getElementById(`view-${viewName}`);
  if (target) {
    target.classList.remove("hidden");
    target.classList.add("active");
  }

  // 타이틀 업데이트
  const titles = {
    building: "건물 현황",
    floor: "층별 현황",
    data: "데이터",
    settings: "설정",
  };
  document.getElementById("view-title").textContent = titles[viewName] || viewName;

  // 층 필터 표시/숨김
  const filterWrap = document.getElementById("floor-filter-wrap");
  filterWrap.style.display = viewName === "building" ? "" : "none";
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
