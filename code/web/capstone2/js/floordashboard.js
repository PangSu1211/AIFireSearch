const FLOOR_DATA = {
  1: {
    totalPeople: 20,
    cameras: [
      { id: '1F-01', x: 4, y: 6, people: 1, trust: 93, status: 'green', place: '입구' },
      { id: '1F-02', x: 26, y: 6, people: 2, trust: 90, status: 'green', place: '회의실 앞' },
      { id: '1F-03', x: 36, y: 6, people: 3, trust: 87, status: 'green', place: '복도' },
      { id: '1F-04', x: 48, y: 6, people: 2, trust: 64, status: 'yellow', place: '사무실 A' },
      { id: '1F-05', x: 66, y: 6, people: 3, trust: 90, status: 'green', place: '우측 복도' },
      { id: '1F-06', x: 4, y: 20, people: 0, trust: 92, status: 'green', place: '좌측 복도' },
      { id: '1F-07', x: 28, y: 48, people: 4, trust: 57, status: 'yellow', place: '사무실 B' },
      { id: '1F-08', x: 46, y: 48, people: 2, trust: 26, status: 'red', place: '중앙 복도' },
      { id: '1F-09', x: 60, y: 48, people: 3, trust: 52, status: 'yellow', place: '우측 사무실' },
      { id: '1F-10', x: 78, y: 48, people: 0, trust: 82, status: 'green', place: '끝 복도' },
    ],
  },
  2: {
    totalPeople: 14,
    cameras: [
      { id: '2F-01', x: 6, y: 8, people: 0, trust: 88, status: 'green', place: '입구' },
      { id: '2F-02', x: 24, y: 10, people: 1, trust: 76, status: 'yellow', place: '복도' },
      { id: '2F-03', x: 47, y: 10, people: 3, trust: 91, status: 'green', place: '사무실' },
      { id: '2F-04', x: 64, y: 34, people: 2, trust: 58, status: 'yellow', place: '회의실' },
      { id: '2F-05', x: 80, y: 52, people: 1, trust: 63, status: 'yellow', place: '출구' },
    ],
  },
  3: {
    totalPeople: 9,
    cameras: [
      { id: '3F-01', x: 10, y: 18, people: 1, trust: 95, status: 'green', place: '좌측' },
      { id: '3F-02', x: 40, y: 22, people: 2, trust: 81, status: 'green', place: '중앙' },
      { id: '3F-03', x: 70, y: 42, people: 0, trust: 58, status: 'yellow', place: '우측' },
    ],
  },
};

let currentFloor = 1;
let selectedCamera = null;

function getStatusText(status) {
  if (status === 'red') return '위험';
  if (status === 'yellow') return '주의';
  return '정상';
}

function renderFloorTabs() {
  const wrap = document.getElementById('floorTabs');
  wrap.innerHTML = '';

  Object.keys(FLOOR_DATA).forEach(floor => {
    const btn = document.createElement('button');
    btn.className = `floor-tab ${Number(floor) === currentFloor ? 'active' : ''}`;
    btn.textContent = `${floor}층`;
    btn.addEventListener('click', () => {
      currentFloor = Number(floor);
      renderAll();
    });
    wrap.appendChild(btn);
  });
}

function renderCameraLayer() {
  const layer = document.getElementById('cameraLayer');
  const floor = FLOOR_DATA[currentFloor];
  layer.innerHTML = '';

  floor.cameras.forEach(camera => {
    const box = document.createElement('button');
    box.type = 'button';
    box.className = `camera-box status-${camera.status}`;
    box.style.left = `${camera.x}%`;
    box.style.top = `${camera.y}%`;
    box.innerHTML = `
      <span class="camera-label">${camera.id}</span>
      <div class="camera-top">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
          <path d="M23 7l-7 5 7 5V7z"></path>
          <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
        </svg>
      </div>
      <div class="camera-bottom">
        <span class="info-badge">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="9" cy="7" r="4"></circle>
          </svg>
          ${camera.people}
        </span>
        <span class="info-badge">
          ${camera.trust}%
        </span>
      </div>
    `;

    box.addEventListener('click', () => {
      selectedCamera = camera;
      updateSelectedPanel();
    });

    layer.appendChild(box);
  });
}

function updateSelectedPanel() {
  const floor = FLOOR_DATA[currentFloor];
  if (!selectedCamera) selectedCamera = floor.cameras[0];

  document.getElementById('selectedId').textContent = selectedCamera.id;
  document.getElementById('selectedMeta').textContent = `${currentFloor}층 ${selectedCamera.place} · 인원 ${selectedCamera.people}명 · 신뢰도 ${selectedCamera.trust}%`;

  const statusEl = document.getElementById('selectedStatus');
  statusEl.textContent = getStatusText(selectedCamera.status);
  statusEl.className = `selected-status ${selectedCamera.status}`;
}

function updateSummary() {
  const floor = FLOOR_DATA[currentFloor];
  const alertCameras = floor.cameras.filter(c => c.status !== 'green');
  const alertCount = alertCameras.length;
  const totalPeople = floor.cameras.reduce((sum, camera) => sum + camera.people, 0);
  const alertPeople = alertCameras.reduce((sum, camera) => sum + camera.people, 0);

  document.getElementById('summary-floor').textContent = `${currentFloor}층`;
  document.getElementById('summary-people').textContent = `${totalPeople}명`;
  document.getElementById('summary-cameras').textContent = `${floor.cameras.length}대`;
  document.getElementById('summary-alerts').textContent = `${alertCount}대`;
  document.getElementById('summary-alert-people').textContent = `${alertPeople}명`;
  document.getElementById('layout-title').textContent = `${currentFloor}층 도면`;
  document.getElementById('layout-subtitle').textContent = `${currentFloor}층 카메라 위치 및 상태 미리보기`;
  document.getElementById('floor-total-badge').textContent = `${totalPeople}명`;
}

function renderAll() {
  renderFloorTabs();
  updateSummary();
  selectedCamera = FLOOR_DATA[currentFloor].cameras[0];
  renderCameraLayer();
  updateSelectedPanel();
}

function updateDateTime() {
  const now = new Date();
  const dateOptions = { year: 'numeric', month: 'short', day: 'numeric' };
  const timeOptions = { hour: '2-digit', minute: '2-digit', hour12: true };

  document.getElementById('current-date').textContent = now.toLocaleDateString('en-US', dateOptions);
  document.getElementById('current-time').textContent = now.toLocaleTimeString('en-US', timeOptions);
}

function applyUrlFloor() {
  const params = new URLSearchParams(window.location.search);
  const floorParam = Number(params.get('floor'));

  if (FLOOR_DATA[floorParam]) {
    currentFloor = floorParam;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  applyUrlFloor();
  updateDateTime();
  setInterval(updateDateTime, 1000);
  renderAll();
});
