// ===== DATA STATE =====
const state = {
  thresholds: [10, 20, 30, 40, 50],
  startDate: { year: 2019, month: 12 },
  endDate: { year: 2023, month: 11 },
  activeCalendar: null, // 'start' | 'end'
  startCalView: { year: 2019, month: 3 },
  endCalView: { year: 2023, month: 12 },
};

const colors = ['#888888', '#4caf50', '#ffc107', '#ff9800', '#f44336'];

// Raw floor population counts (fixed)
const floorData = [78, 69, 20, 45, 12, 61, 40, 46, 21];
const floors = ['1층','2층','3층','4층','5층','6층','7층','8층','9층'];

// Hourly: per floor, per time segment
const timeSegments = ['0시~8시','8시~11시','11시~13시','13시~18시','18시~0시'];
const hourlyRaw = [
  [5, 22, 35, 28, 8],   // 9층
  [6, 25, 38, 30, 9],   // 8층
  [4, 18, 27, 22, 6],   // 7층
  [7, 28, 42, 35, 10],  // 6층
  [3, 15, 22, 18, 5],   // 5층
  [8, 32, 45, 38, 12],  // 4층
  [6, 20, 30, 25, 8],   // 3층
  [9, 35, 48, 40, 14],  // 2층
  [5, 18, 25, 20, 7],   // 1층
];

let hourlyChart = null;
let floorChart = null;
let expandedChartInstance = null;
let donutChart = null;

// ===== UTILS =====
function getColor(val) {
  const t = state.thresholds;
  if (val <= t[0]) return colors[0];
  if (val <= t[1]) return colors[1];
  if (val <= t[2]) return colors[2];
  if (val <= t[3]) return colors[3];
  return colors[4];
}

function segmentColor(val) {
  return getColor(val);
}

// ===== HOURLY CHART =====
function buildHourlyDatasets() {
  return timeSegments.map((seg, si) => ({
    label: seg,
    data: hourlyRaw.map(row => row[si]),
    backgroundColor: hourlyRaw.map(row => segmentColor(row[si])),
    borderWidth: 0,
    borderRadius: 3,
  }));
}

function initHourlyChart(canvasId) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (canvasId === 'hourlyChart' && hourlyChart) hourlyChart.destroy();
  if (canvasId === 'expandedChart' && expandedChartInstance) expandedChartInstance.destroy();

  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['9층','8층','7층','6층','5층','4층','3층','2층','1층'],
      datasets: buildHourlyDatasets(),
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { mode: 'index' } },
      scales: {
        x: { stacked: true, ticks: { font: { size: 11 } }, grid: { display: false } },
        y: { stacked: true, ticks: { font: { size: 11 } }, grid: { display: false } },
      },
    },
  });

  if (canvasId === 'hourlyChart') hourlyChart = chart;
  else expandedChartInstance = chart;
  return chart;
}

// ===== FLOOR CHART =====
function initFloorChart() {
  const ctx = document.getElementById('floorChart').getContext('2d');
  if (floorChart) floorChart.destroy();
  floorChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: floors,
      datasets: [{
        data: floorData,
        backgroundColor: '#5b6af0',
        borderRadius: 5,
        borderWidth: 0,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        datalabels: { display: false },
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { grid: { color: '#f0f0f0' }, ticks: { font: { size: 11 } } },
      },
    },
    plugins: [{
      id: 'dataLabels',
      afterDatasetsDraw(chart) {
        const { ctx, data } = chart;
        ctx.save();
        ctx.font = '11px sans-serif';
        ctx.fillStyle = '#444';
        ctx.textAlign = 'center';
        chart.getDatasetMeta(0).data.forEach((bar, i) => {
          ctx.fillText(data.datasets[0].data[i], bar.x, bar.y - 4);
        });
        ctx.restore();
      }
    }],
  });
}

// ===== DONUT CHART =====
function initDonutChart() {
  const ctx = document.getElementById('donutChart').getContext('2d');
  if (donutChart) donutChart.destroy();
  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [75, 25],
        backgroundColor: ['#5b6af0', '#e5e7eb'],
        borderWidth: 0,
      }],
    },
    options: {
      cutout: '68%',
      responsive: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
}

// ===== MODAL HELPERS =====
function openModal(id) {
  document.getElementById(id).classList.add('open');
}
function closeModal(id) {
  document.getElementById(id).classList.remove('open');
}

// Close buttons
document.querySelectorAll('.close-btn, .btn-cancel').forEach(btn => {
  const target = btn.dataset.close;
  if (target) btn.addEventListener('click', () => closeModal(target));
});

// Click outside to close
document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', e => {
    if (e.target === overlay) closeModal(overlay.id);
  });
});

// ===== ⓐ CHART EXPAND =====
function openChartExpand(title, type) {
  document.getElementById('chartModalTitle').textContent = title;
  openModal('chartModal');
  setTimeout(() => {
    const canvas = document.getElementById('expandedChart');
    canvas.width = canvas.parentElement.offsetWidth;
    canvas.height = 400;
    if (type === 'hourly') {
      initHourlyChart('expandedChart');
    } else {
      const ctx = canvas.getContext('2d');
      if (expandedChartInstance) expandedChartInstance.destroy();
      expandedChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: floors,
          datasets: [{
            data: floorData,
            backgroundColor: '#5b6af0',
            borderRadius: 5, borderWidth: 0,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { grid: { display: false } },
            y: { grid: { color: '#f0f0f0' } },
          },
        },
      });
    }
  }, 100);
}

document.getElementById('hourlyChartWrap').addEventListener('click', () => openChartExpand('시간별 인원분포도', 'hourly'));
document.getElementById('floorChartWrap').addEventListener('click', () => openChartExpand('층별 인원분포도', 'floor'));

// ===== ⓑ CAMERA CHECK =====
const checkBadgeBtn = document.getElementById('checkBadgeBtn');
const checkList = document.getElementById('checkList');

checkBadgeBtn.addEventListener('click', () => {
  checkBadgeBtn.classList.toggle('open');
  checkList.classList.toggle('visible');
});

const fireVideoData = {
  video1: { title: '2023.4.6 - 2층 구내식당', date: '2023년 4월 6일', location: '2층 구내식당', count: 21, trust: 42 },
  video2: { title: '2019.12.3 - 3층 304호', date: '2019년 12월 3일', location: '3층 304호', count: 2, trust: 21 },
};

document.querySelectorAll('.check-item').forEach(item => {
  item.addEventListener('click', () => {
    const cam = item.dataset.cam;
    document.getElementById('cameraModalTitle').textContent = `카메라 점검 — ${cam}`;
    document.getElementById('cameraLabel').textContent = cam;
    document.getElementById('camInfoText').innerHTML = `카메라: <strong>${cam}</strong>`;
    openModal('cameraModal');
  });
});

// ===== ⓒ BELL NOTIFICATION =====
document.getElementById('bellBtn').addEventListener('click', () => openModal('notifModal'));

// Notification actions
document.querySelectorAll('.notif-confirm-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const action = btn.dataset.action;
    const target = btn.dataset.target;
    closeModal('notifModal');
    if (action === 'navigate' && target) {
      const el = document.querySelector(`.camera-card, .fire-card`);
      if (target === 'cameraSection') {
        document.querySelector('.camera-card').scrollIntoView({ behavior: 'smooth' });
        document.querySelector('.camera-card').style.outline = '2px solid var(--accent)';
        setTimeout(() => document.querySelector('.camera-card').style.outline = '', 2000);
      } else if (target === 'fireSection') {
        document.querySelector('.fire-card').scrollIntoView({ behavior: 'smooth' });
        document.querySelector('.fire-card').style.outline = '2px solid var(--red)';
        setTimeout(() => document.querySelector('.fire-card').style.outline = '', 2000);
      }
    }
  });
});

document.querySelectorAll('.notif-x').forEach(btn => {
  btn.addEventListener('click', () => {
    btn.closest('.notif-item').style.opacity = '0';
    btn.closest('.notif-item').style.transform = 'translateX(20px)';
    btn.closest('.notif-item').style.transition = 'all .3s';
    setTimeout(() => btn.closest('.notif-item').remove(), 300);
  });
});

// ===== ⓓ VIDEO POPUP =====
document.querySelectorAll('.video-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const vid = btn.dataset.video;
    const data = fireVideoData[vid];
    if (!data) return;
    document.getElementById('videoModalTitle').textContent = `화재 의심 영상 — ${data.date}`;
    document.getElementById('videoLabel').textContent = `${data.location} 화재 의심 영상`;
    document.getElementById('videoInfo').innerHTML = `
      <p>일시: <strong>${data.date}</strong></p>
      <p>장소: <strong>${data.location}</strong></p>
      <p>인원 수: <strong>${data.count}명</strong></p>
      <p>카메라 신뢰도: <strong>${data.trust}</strong></p>
    `;
    openModal('videoModal');
  });
});

// ===== ⓕ SETTINGS =====
document.getElementById('gearBtn').addEventListener('click', () => {
  updateRangeInputs();
  openModal('settingsModal');
});

function updateRangeInputs() {
  document.querySelectorAll('.range-input').forEach((input, i) => {
    input.value = state.thresholds[i];
  });
  document.getElementById('startDateInput').value = `${state.startDate.year}.${String(state.startDate.month).padStart(2,'0')}`;
  document.getElementById('endDateInput').value = `${state.endDate.year}.${String(state.endDate.month).padStart(2,'0')}`;
}

// Calendar toggle
const startInput = document.getElementById('startDateInput');
const endInput = document.getElementById('endDateInput');
const calendarsRow = document.getElementById('calendarsRow');

function showCalendars(which) {
  state.activeCalendar = which;
  calendarsRow.style.display = 'flex';
  renderBothCals();
}

startInput.addEventListener('click', () => showCalendars('start'));
endInput.addEventListener('click', () => showCalendars('end'));

// Calendar rendering
function monthName(year, month) {
  return `${year}.${String(month).padStart(2,'0')}`;
}

function renderCal(side) {
  const view = side === 'start' ? state.startCalView : state.endCalView;
  const labelEl = document.getElementById(`${side}MonthLabel`);
  const gridEl = document.getElementById(`${side}CalGrid`);
  labelEl.textContent = monthName(view.year, view.month);
  gridEl.innerHTML = '';

  const daysInMonth = new Date(view.year, view.month, 0).getDate();
  const firstDay = new Date(view.year, view.month - 1, 1).getDay();

  for (let i = 0; i < firstDay; i++) {
    const empty = document.createElement('div');
    empty.className = 'cal-day empty';
    gridEl.appendChild(empty);
  }
  for (let d = 1; d <= daysInMonth; d++) {
    const day = document.createElement('div');
    day.className = 'cal-day';
    day.textContent = d;

    const isSelected = (
      side === 'start' &&
      state.startDate.year === view.year &&
      state.startDate.month === view.month &&
      state.startDate.day === d
    ) || (
      side === 'end' &&
      state.endDate.year === view.year &&
      state.endDate.month === view.month &&
      state.endDate.day === d
    );
    if (isSelected) day.classList.add('selected');

    day.addEventListener('click', () => {
      if (state.activeCalendar === 'start') {
        state.startDate = { year: view.year, month: view.month, day: d };
        startInput.value = `${view.year}.${String(view.month).padStart(2,'0')}`;
      } else {
        state.endDate = { year: view.year, month: view.month, day: d };
        endInput.value = `${view.year}.${String(view.month).padStart(2,'0')}`;
      }
      renderBothCals();
    });
    gridEl.appendChild(day);
  }
}

function renderBothCals() {
  renderCal('start');
  renderCal('end');
}

// Calendar nav buttons
document.getElementById('startPrev').addEventListener('click', () => {
  state.startCalView.month--;
  if (state.startCalView.month < 1) { state.startCalView.month = 12; state.startCalView.year--; }
  renderCal('start');
});
document.getElementById('startNext').addEventListener('click', () => {
  state.startCalView.month++;
  if (state.startCalView.month > 12) { state.startCalView.month = 1; state.startCalView.year++; }
  renderCal('start');
});
document.getElementById('endPrev').addEventListener('click', () => {
  state.endCalView.month--;
  if (state.endCalView.month < 1) { state.endCalView.month = 12; state.endCalView.year--; }
  renderCal('end');
});
document.getElementById('endNext').addEventListener('click', () => {
  state.endCalView.month++;
  if (state.endCalView.month > 12) { state.endCalView.month = 1; state.endCalView.year++; }
  renderCal('end');
});

// Apply settings
document.getElementById('applySettings').addEventListener('click', () => {
  const inputs = document.querySelectorAll('.range-input');
  const newThresholds = [];
  let valid = true;
  inputs.forEach((input, i) => {
    const val = parseInt(input.value);
    if (isNaN(val) || val < 1) { valid = false; return; }
    newThresholds.push(val);
  });
  if (!valid) { alert('올바른 인원 수를 입력해주세요.'); return; }
  state.thresholds = newThresholds;

  // Update legend in main page
  const labels = [
    `${newThresholds[0]}명 이하`,
    `${newThresholds[0]+1}~${newThresholds[1]}명`,
    `${newThresholds[1]+1}~${newThresholds[2]}명`,
    `${newThresholds[2]+1}~${newThresholds[3]}명`,
    `${newThresholds[3]+1}명 이상`,
  ];
  document.querySelectorAll('.leg-item').forEach((item, i) => {
    const textNode = item.childNodes[1];
    if (textNode) textNode.textContent = labels[i];
  });

  // Rebuild hourly chart with new colors
  initHourlyChart('hourlyChart');
  closeModal('settingsModal');
  calendarsRow.style.display = 'none';
});

// ===== INIT =====
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('hourlyChart').parentElement.style.height = '220px';
  document.getElementById('floorChart').parentElement.style.height = '200px';
  initHourlyChart('hourlyChart');
  initFloorChart();
  initDonutChart();
  updateRangeInputs();
});

function updateDateTime(){

  const now = new Date();

  const dateOptions = {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  };

  const timeOptions = {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true
  };

  const currentDate =
    now.toLocaleDateString('ko-KR', dateOptions);

  const currentTime =
    now.toLocaleTimeString('ko-KR', timeOptions);

  document.getElementById('current-date').textContent =
    currentDate;

  document.getElementById('current-time').textContent =
    currentTime;
}

/* 실행 */
updateDateTime();

/* 1초마다 갱신 */
setInterval(updateDateTime, 1000);
