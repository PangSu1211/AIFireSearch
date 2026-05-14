"use strict";

/* ============================================================
   Floor Dashboard — Real Floor Plans (1F / 2F / 3F)
============================================================ */

const BUILDING_CONFIG_KEY = "buildingConfig";
const CAMERA_LAYOUT_KEY   = "cameraLayout";
const AI_RESULTS_KEY      = "aiCameraResults";

/* ===== DEFAULT AI DATA ===== */
const DEFAULT_AI_RESULTS = [
  { id: "1F-01", floor: 1, people: 1,  confidence: 93 },
  { id: "1F-02", floor: 1, people: 2,  confidence: 90 },
  { id: "1F-03", floor: 1, people: 3,  confidence: 87 },
  { id: "1F-04", floor: 1, people: 0,  confidence: 64 },
  { id: "1F-05", floor: 1, people: 3,  confidence: 26 },
  { id: "1F-06", floor: 1, people: 2,  confidence: 92 },

  { id: "2F-01", floor: 2, people: 8,  confidence: 88 },
  { id: "2F-02", floor: 2, people: 5,  confidence: 57 },
  { id: "2F-03", floor: 2, people: 3,  confidence: 91 },
  { id: "2F-04", floor: 2, people: 2,  confidence: 58 },
  { id: "2F-05", floor: 2, people: 4,  confidence: 76 },
  { id: "2F-06", floor: 2, people: 1,  confidence: 63 },

  { id: "3F-01", floor: 3, people: 6,  confidence: 95 },
  { id: "3F-02", floor: 3, people: 4,  confidence: 81 },
  { id: "3F-03", floor: 3, people: 2,  confidence: 58 },
  { id: "3F-04", floor: 3, people: 3,  confidence: 42 },
  { id: "3F-05", floor: 3, people: 1,  confidence: 89 },
  { id: "3F-06", floor: 3, people: 5,  confidence: 72 },
  { id: "3F-07", floor: 3, people: 0,  confidence: 93 },
];

/* ===== FLOOR PLAN DEFINITIONS =====
   viewBox: "0 0 W H"
   cameras: { id, floor, place, cx, cy, w, h }  — all in SVG units
================================================================ */
const FLOOR_PLANS = {

  /* ── 1층 ─────────────────────────────────────────────── */
  1: {
    viewBox: "0 0 1000 760",
    svgDefs: `
      <!-- outer wall -->
      <rect x="20" y="20" width="960" height="620" fill="#f2f2f0" stroke="#4a4a4a" stroke-width="10" rx="2"/>

      <!-- 출입구 하단 돌출 -->
      <rect x="310" y="610" width="380" height="90" fill="#e8e8e6" stroke="#4a4a4a" stroke-width="7"/>
      <line x1="390" y1="612" x2="390" y2="698" stroke="#888" stroke-width="3"/>
      <line x1="500" y1="612" x2="500" y2="698" stroke="#888" stroke-width="3"/>
      <line x1="610" y1="612" x2="610" y2="698" stroke="#888" stroke-width="3"/>
      <path d="M460 612 Q500 680 540 612" fill="none" stroke="#888" stroke-width="2"/>
      <text x="500" y="665" text-anchor="middle" font-size="15" fill="#777" font-weight="600">출입구</text>

      <!-- 엘리베이터 룸 (좌상) -->
      <rect x="20" y="20" width="230" height="250" fill="#e3e3e0" stroke="#4a4a4a" stroke-width="7"/>
      <rect x="45"  y="45"  width="82" height="82" fill="#c8c8c4" stroke="#888" stroke-width="3"/>
      <rect x="148" y="45"  width="82" height="82" fill="#c8c8c4" stroke="#888" stroke-width="3"/>
      <line x1="45"  y1="45"  x2="127" y2="127" stroke="#aaa" stroke-width="2"/>
      <line x1="127" y1="45"  x2="45"  y2="127" stroke="#aaa" stroke-width="2"/>
      <line x1="148" y1="45"  x2="230" y2="127" stroke="#aaa" stroke-width="2"/>
      <line x1="230" y1="45"  x2="148" y2="127" stroke="#aaa" stroke-width="2"/>
      <path d="M230 160 A40 40 0 0 1 190 200" fill="none" stroke="#888" stroke-width="2.5"/>
      <text x="128" y="216" text-anchor="middle" font-size="15" fill="#555" font-weight="600">엘리베이터</text>

      <!-- 보안실 (좌하) -->
      <rect x="20" y="450" width="180" height="190" fill="#e3e3e0" stroke="#4a4a4a" stroke-width="7"/>
      <rect x="42" y="500" width="90" height="55" fill="#c8c8c4" stroke="#888" stroke-width="2"/>
      <rect x="55" y="510" width="25" height="15" fill="#aaa" stroke="#999" stroke-width="1"/>
      <rect x="85" y="510" width="25" height="15" fill="#aaa" stroke="#999" stroke-width="1"/>
      <path d="M200 450 A40 40 0 0 0 200 490" fill="none" stroke="#888" stroke-width="2.5" transform="rotate(90,200,470)"/>
      <text x="108" y="488" text-anchor="middle" font-size="15" fill="#555" font-weight="600">보안실</text>

      <!-- 계단 (우상) -->
      <rect x="800" y="20" width="180" height="210" fill="#e3e3e0" stroke="#4a4a4a" stroke-width="7"/>
      <line x1="800" y1="55"  x2="980" y2="55"  stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="85"  x2="980" y2="85"  stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="115" x2="980" y2="115" stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="145" x2="980" y2="145" stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="175" x2="980" y2="175" stroke="#999" stroke-width="2.5"/>
      <text x="890" y="205" text-anchor="middle" font-size="15" fill="#555" font-weight="600">계단</text>

      <!-- 안내데스크 (상중) -->
      <ellipse cx="505" cy="135" rx="150" ry="75" fill="#d8d8d4" stroke="#888" stroke-width="3"/>
      <ellipse cx="505" cy="135" rx="110" ry="50" fill="#c8c8c4" stroke="#aaa" stroke-width="2"/>
      <rect x="435" y="115" width="40" height="30" fill="#bbb" rx="3"/>
      <rect x="540" y="115" width="40" height="30" fill="#bbb" rx="3"/>
      <text x="505" y="235" text-anchor="middle" font-size="15" fill="#555" font-weight="600">안내데스크</text>

      <!-- 로비 레이블 -->
      <text x="460" y="430" text-anchor="middle" font-size="32" fill="#c8c8c4" font-weight="700" letter-spacing="6">로비</text>

      <!-- 대기공간 소파 (우하) -->
      <rect x="720" y="360" width="260" height="220" fill="#ebebea" stroke="#aaa" stroke-width="1.5" stroke-dasharray="7,4"/>
      <rect x="738" y="375" width="75" height="55" fill="#d2d2ce" stroke="#999" stroke-width="2" rx="5"/>
      <rect x="826" y="375" width="75" height="55" fill="#d2d2ce" stroke="#999" stroke-width="2" rx="5"/>
      <rect x="738" y="450" width="210" height="65" fill="#d2d2ce" stroke="#999" stroke-width="2" rx="5"/>
      <rect x="738" y="520" width="210" height="28" fill="#c5c5c1" stroke="#aaa" stroke-width="1.5" rx="3"/>
      <text x="840" y="578" text-anchor="middle" font-size="14" fill="#777" font-weight="600">대기공간</text>
    `,
    cameras: [
      { id:"1F-01", floor:1, place:"엘리베이터",  cx:128, cy:150, w:160, h:100 },
      { id:"1F-02", floor:1, place:"보안실",       cx:110, cy:520, w:140, h:95  },
      { id:"1F-03", floor:1, place:"안내데스크",   cx:505, cy:140, w:190, h:100 },
      { id:"1F-04", floor:1, place:"로비 중앙",    cx:450, cy:400, w:180, h:100 },
      { id:"1F-05", floor:1, place:"대기공간",     cx:840, cy:460, w:160, h:100 },
      { id:"1F-06", floor:1, place:"출입구",       cx:500, cy:640, w:180, h:90  },
    ],
  },

  /* ── 2층 ─────────────────────────────────────────────── */
  2: {
    viewBox: "0 0 1000 760",
    svgDefs: `
      <!-- outer wall -->
      <rect x="20" y="20" width="960" height="720" fill="#f2f2f0" stroke="#4a4a4a" stroke-width="10" rx="2"/>

      <!-- 주방 구역 (좌상) -->
      <rect x="20" y="20" width="460" height="235" fill="#e3e3e0" stroke="#4a4a4a" stroke-width="7"/>
      <!-- 창고 -->
      <rect x="20" y="20" width="115" height="130" fill="#d8d8d4" stroke="#666" stroke-width="5"/>
      <text x="77" y="90" text-anchor="middle" font-size="13" fill="#666" font-weight="600">창고</text>
      <!-- 주방 기기 -->
      <rect x="145" y="30" width="200" height="55" fill="#c8c8c4" stroke="#999" stroke-width="2" rx="3"/>
      <rect x="355" y="30" width="115" height="55" fill="#c8c8c4" stroke="#999" stroke-width="2" rx="3"/>
      <rect x="145" y="95" width="90"  height="55" fill="#c8c8c4" stroke="#999" stroke-width="2" rx="3"/>
      <text x="260" y="90" text-anchor="middle" font-size="17" fill="#666" font-weight="600">주방</text>
      <!-- 배식구 -->
      <rect x="330" y="140" width="140" height="105" fill="#d0d0cc" stroke="#888" stroke-width="3"/>
      <text x="400" y="200" text-anchor="middle" font-size="14" fill="#666" font-weight="600">배식구</text>
      <!-- 퇴식구 -->
      <rect x="455" y="440" width="95" height="95" fill="#d0d0cc" stroke="#888" stroke-width="3"/>
      <text x="502" y="494" text-anchor="middle" font-size="12" fill="#666" font-weight="600">퇴식구</text>

      <!-- 구내식당 (좌 메인) -->
      <rect x="20" y="255" width="510" height="465" fill="#eeeeec" stroke="#4a4a4a" stroke-width="7"/>
      <text x="240" y="500" text-anchor="middle" font-size="24" fill="#c0c0bc" font-weight="700" letter-spacing="3">구내식당</text>
      <!-- 테이블 그룹들 -->
      <rect x="42"  y="278" width="120" height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="42"  y="358" width="120" height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="42"  y="438" width="120" height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="42"  y="518" width="120" height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="42"  y="598" width="120" height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="42"  y="678" width="120" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="205" y="315" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="320" y="315" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="205" y="425" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="320" y="425" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="205" y="640" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="320" y="640" width="90"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>

      <!-- 계단 (중앙) -->
      <rect x="530" y="310" width="140" height="240" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <line x1="530" y1="348" x2="670" y2="348" stroke="#999" stroke-width="2.5"/>
      <line x1="530" y1="378" x2="670" y2="378" stroke="#999" stroke-width="2.5"/>
      <line x1="530" y1="408" x2="670" y2="408" stroke="#999" stroke-width="2.5"/>
      <line x1="530" y1="438" x2="670" y2="438" stroke="#999" stroke-width="2.5"/>
      <line x1="530" y1="468" x2="670" y2="468" stroke="#999" stroke-width="2.5"/>
      <text x="600" y="564" text-anchor="middle" font-size="15" fill="#555" font-weight="600">계단</text>

      <!-- 엘리베이터 (중상) -->
      <rect x="530" y="20" width="140" height="195" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <rect x="548" y="38"  width="48" height="65" fill="#c8c8c4" stroke="#888" stroke-width="2"/>
      <rect x="608" y="38"  width="48" height="65" fill="#c8c8c4" stroke="#888" stroke-width="2"/>
      <line x1="548" y1="38"  x2="596" y2="103" stroke="#aaa" stroke-width="1.5"/>
      <line x1="596" y1="38"  x2="548" y2="103" stroke="#aaa" stroke-width="1.5"/>
      <line x1="608" y1="38"  x2="656" y2="103" stroke="#aaa" stroke-width="1.5"/>
      <line x1="656" y1="38"  x2="608" y2="103" stroke="#aaa" stroke-width="1.5"/>
      <text x="600" y="195" text-anchor="middle" font-size="13" fill="#555" font-weight="600">엘리베이터</text>

      <!-- 화장실 (하중) -->
      <rect x="530" y="620" width="140" height="120" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="6"/>
      <rect x="530" y="620" width="70"  height="120" fill="#d8d8d4" stroke="#888" stroke-width="3"/>
      <text x="565" y="688" text-anchor="middle" font-size="12" fill="#666">남자</text>
      <text x="635" y="688" text-anchor="middle" font-size="12" fill="#666">여자</text>

      <!-- 구내카페 (우) -->
      <rect x="670" y="20" width="310" height="720" fill="#eeeeec" stroke="#4a4a4a" stroke-width="7"/>
      <!-- 창고 우상 -->
      <rect x="840" y="20" width="140" height="120" fill="#d8d8d4" stroke="#666" stroke-width="5"/>
      <text x="910" y="88" text-anchor="middle" font-size="13" fill="#666" font-weight="600">창고</text>
      <!-- 카운터 -->
      <rect x="678" y="35"  width="90" height="110" fill="#c8c8c4" stroke="#888" stroke-width="3" rx="4"/>
      <text x="723" y="162" text-anchor="middle" font-size="13" fill="#666" font-weight="600">카운터</text>
      <!-- 카페 레이블 -->
      <text x="825" y="430" text-anchor="middle" font-size="22" fill="#c0c0bc" font-weight="700" letter-spacing="2">구내 카페</text>
      <!-- 카페 테이블 -->
      <rect x="690" y="210" width="80"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="800" y="210" width="65"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="890" y="210" width="65"  height="60" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="690" y="320" width="130" height="75" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="850" y="320" width="90"  height="75" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="690" y="450" width="130" height="65" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="850" y="450" width="90"  height="65" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="690" y="580" width="270" height="65" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="690" y="670" width="270" height="55" fill="#d2d2ce" stroke="#aaa" stroke-width="2" rx="3"/>

      <!-- 문 arc -->
      <path d="M530 258 A40 40 0 0 0 490 298" fill="none" stroke="#888" stroke-width="2.5"/>
      <path d="M670 215 A40 40 0 0 1 710 255" fill="none" stroke="#888" stroke-width="2.5"/>
    `,
    cameras: [
      { id:"2F-01", floor:2, place:"구내식당 좌",    cx:235, cy:460, w:170, h:105 },
      { id:"2F-02", floor:2, place:"주방",            cx:225, cy:115, w:170, h:100 },
      { id:"2F-03", floor:2, place:"구내식당 우",     cx:420, cy:600, w:160, h:100 },
      { id:"2F-04", floor:2, place:"계단/엘리베이터", cx:600, cy:240, w:150, h:100 },
      { id:"2F-05", floor:2, place:"구내카페 상",     cx:830, cy:260, w:165, h:100 },
      { id:"2F-06", floor:2, place:"구내카페 하",     cx:830, cy:530, w:165, h:100 },
    ],
  },

  /* ── 3층 ─────────────────────────────────────────────── */
  3: {
    viewBox: "0 0 1000 720",
    svgDefs: `
      <!-- outer wall -->
      <rect x="20" y="20" width="960" height="680" fill="#f2f2f0" stroke="#4a4a4a" stroke-width="10" rx="2"/>

      <!-- 복도 (수평 중앙) -->
      <rect x="20" y="330" width="780" height="70" fill="#e8e8e6" stroke="none"/>
      <text x="400" y="372" text-anchor="middle" font-size="14" fill="#aaa" font-weight="600" letter-spacing="2">복도</text>

      <!-- 사무실 A (좌상) -->
      <rect x="20" y="20" width="445" height="310" fill="#eeeeec" stroke="#4a4a4a" stroke-width="7"/>
      <text x="222" y="210" text-anchor="middle" font-size="20" fill="#c0c0bc" font-weight="700">사무실 A</text>
      <rect x="48"  y="48"  width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="122" y="48"  width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="196" y="48"  width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="270" y="48"  width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="344" y="48"  width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="48"  y="110" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="122" y="110" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="196" y="110" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="270" y="110" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="344" y="110" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="48"  y="248" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="122" y="248" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="196" y="248" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="270" y="248" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <path d="M465 290 A45 45 0 0 0 420 330" fill="none" stroke="#888" stroke-width="2.5"/>

      <!-- 회의실 (상중) -->
      <rect x="465" y="20" width="195" height="310" fill="#e5e5e2" stroke="#4a4a4a" stroke-width="7"/>
      <text x="562" y="215" text-anchor="middle" font-size="16" fill="#888" font-weight="600">회의실</text>
      <rect x="490" y="48"  width="145" height="230" fill="#d2d2ce" stroke="#aaa" stroke-width="2" rx="4"/>
      <path d="M660 290 A45 45 0 0 1 705 330" fill="none" stroke="#888" stroke-width="2.5"/>

      <!-- 탕비실 (상우중) -->
      <rect x="660" y="20" width="140" height="310" fill="#e5e5e2" stroke="#4a4a4a" stroke-width="7"/>
      <text x="730" y="215" text-anchor="middle" font-size="14" fill="#888" font-weight="600">탕비실</text>
      <rect x="670" y="35"  width="120" height="42" fill="#c8c8c4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="670" y="225" width="120" height="55" fill="#c8c8c4" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="670" y="288" width="120" height="38" fill="#c8c8c4" stroke="#aaa" stroke-width="2" rx="3"/>

      <!-- 계단 (우상) -->
      <rect x="800" y="20" width="180" height="215" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <line x1="800" y1="58"  x2="980" y2="58"  stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="90"  x2="980" y2="90"  stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="122" x2="980" y2="122" stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="154" x2="980" y2="154" stroke="#999" stroke-width="2.5"/>
      <line x1="800" y1="186" x2="980" y2="186" stroke="#999" stroke-width="2.5"/>
      <text x="890" y="222" text-anchor="middle" font-size="14" fill="#555" font-weight="600">계단</text>

      <!-- 엘리베이터 (우중) -->
      <rect x="800" y="235" width="180" height="165" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <rect x="818" y="252" width="65" height="85" fill="#c8c8c4" stroke="#888" stroke-width="2"/>
      <line x1="818" y1="252" x2="883" y2="337" stroke="#aaa" stroke-width="1.5"/>
      <line x1="883" y1="252" x2="818" y2="337" stroke="#aaa" stroke-width="1.5"/>
      <text x="890" y="342" text-anchor="middle" font-size="12" fill="#555" font-weight="600">엘리베이터</text>

      <!-- 남자/여자 화장실 (우하) -->
      <rect x="800" y="400" width="180" height="300" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <rect x="800" y="400" width="180" height="150" fill="#d8d8d4" stroke="#888" stroke-width="3"/>
      <text x="890" y="480" text-anchor="middle" font-size="12" fill="#666" font-weight="600">남자 화장실</text>
      <text x="890" y="600" text-anchor="middle" font-size="12" fill="#666" font-weight="600">여자 화장실</text>

      <!-- 임원실 (좌하) -->
      <rect x="20" y="400" width="200" height="300" fill="#e5e5e2" stroke="#4a4a4a" stroke-width="7"/>
      <text x="120" y="500" text-anchor="middle" font-size="16" fill="#888" font-weight="600">임원실</text>
      <rect x="42"  y="418" width="130" height="85"  fill="#d2d2ce" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="38"  y="560" width="85"  height="55"  fill="#d2d2ce" stroke="#aaa" stroke-width="2" rx="3"/>
      <rect x="140" y="562" width="60"  height="42"  fill="#d2d2ce" stroke="#aaa" stroke-width="2" rx="3"/>
      <path d="M220 430 A45 45 0 0 1 220 400" fill="none" stroke="#888" stroke-width="2.5" transform="rotate(180,220,415)"/>

      <!-- 사무실 B (하중) -->
      <rect x="220" y="400" width="440" height="300" fill="#eeeeec" stroke="#4a4a4a" stroke-width="7"/>
      <text x="440" y="555" text-anchor="middle" font-size="20" fill="#c0c0bc" font-weight="700">사무실 B</text>
      <rect x="248" y="425" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="322" y="425" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="396" y="425" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="470" y="425" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="544" y="425" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="248" y="483" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="322" y="483" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="396" y="483" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="470" y="483" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="544" y="483" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="248" y="598" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="322" y="598" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="396" y="598" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="470" y="598" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <rect x="544" y="598" width="62" height="42" fill="#d8d8d4" stroke="#aaa" stroke-width="2" rx="2"/>
      <path d="M660 430 A45 45 0 0 0 660 400" fill="none" stroke="#888" stroke-width="2.5" transform="rotate(90,660,415)"/>

      <!-- 서버실 (하우) -->
      <rect x="660" y="400" width="140" height="300" fill="#e0e0dd" stroke="#4a4a4a" stroke-width="7"/>
      <text x="730" y="548" text-anchor="middle" font-size="14" fill="#888" font-weight="600">서버실</text>
      <rect x="674" y="418" width="112" height="272" fill="#ccccca" stroke="#aaa" stroke-width="2" rx="3"/>
      <line x1="674" y1="458" x2="786" y2="458" stroke="#bbb" stroke-width="2"/>
      <line x1="674" y1="498" x2="786" y2="498" stroke="#bbb" stroke-width="2"/>
      <line x1="674" y1="538" x2="786" y2="538" stroke="#bbb" stroke-width="2"/>
      <line x1="674" y1="578" x2="786" y2="578" stroke="#bbb" stroke-width="2"/>
      <line x1="674" y1="618" x2="786" y2="618" stroke="#bbb" stroke-width="2"/>
      <line x1="674" y1="658" x2="786" y2="658" stroke="#bbb" stroke-width="2"/>
    `,
    cameras: [
      { id:"3F-01", floor:3, place:"사무실 A",  cx:222, cy:155, w:175, h:105 },
      { id:"3F-02", floor:3, place:"회의실",    cx:562, cy:155, w:160, h:105 },
      { id:"3F-03", floor:3, place:"탕비실",    cx:730, cy:155, w:130, h:100 },
      { id:"3F-04", floor:3, place:"임원실",    cx:120, cy:510, w:155, h:105 },
      { id:"3F-05", floor:3, place:"사무실 B",  cx:440, cy:510, w:175, h:105 },
      { id:"3F-06", floor:3, place:"서버실",    cx:730, cy:548, w:130, h:100 },
      { id:"3F-07", floor:3, place:"복도",      cx:390, cy:365, w:175, h:65  },
    ],
  },
};

/* ===== STATE ===== */
let FLOOR_DATA     = {};
let currentFloor   = 1;
let selectedCamera = null;
let initialCameraId = null;
let shouldOpenInitialCamera = false;

/* ===== UTILS ===== */
function safeJsonParse(v, fallback) {
  try { return v ? JSON.parse(v) : fallback; } catch { return fallback; }
}
function readThresholds() {
  try { const t = localStorage.getItem("adminAlertThresholds"); if (t) return JSON.parse(t); } catch {}
  return { greenMin: 80, yellowMin: 50 };
}
function getStatus(trust) {
  const t = readThresholds();
  if (trust >= t.greenMin) return "green";
  if (trust >= t.yellowMin) return "yellow";
  return "red";
}
function statusText(s) {
  return s === "red" ? "위험" : s === "yellow" ? "주의" : "정상";
}

/* ===== DATA ===== */
function normalizeAI(r) {
  return { id:String(r.id), floor:Number(r.floor), people:Number(r.people??0), confidence:Number(r.confidence??r.trust??100) };
}
function readAIResults() {
  const adminCams = safeJsonParse(localStorage.getItem("adminCameras"), null);
  if (Array.isArray(adminCams) && adminCams.length > 0) {
    return adminCams.map(c => normalizeAI({ id:c.id, floor:c.floor, people:c.people, confidence:c.confidence }));
  }
  const saved = safeJsonParse(localStorage.getItem(AI_RESULTS_KEY), null);
  return Array.isArray(saved) ? saved.map(normalizeAI) : DEFAULT_AI_RESULTS.map(normalizeAI);
}
function buildFloorData() {
  const aiResults = readAIResults();
  const resultMap = new Map(aiResults.map(r => [r.id, r]));
  FLOOR_DATA = {};
  Object.entries(FLOOR_PLANS).forEach(([floorNum, plan]) => {
    const fl = Number(floorNum);
    FLOOR_DATA[fl] = {
      cameras: plan.cameras.map(cam => {
        const res = resultMap.get(cam.id);
        return { ...cam, people:Number(res?.people??0), trust:Number(res?.confidence??100) };
      }),
    };
  });
}

/* Public API */
function setCameraAIResults(results) { localStorage.setItem(AI_RESULTS_KEY, JSON.stringify(results)); refresh(); }
function resetDashboardDefaults()    { localStorage.removeItem(AI_RESULTS_KEY); location.reload(); }
window.setCameraAIResults    = setCameraAIResults;
window.resetDashboardDefaults = resetDashboardDefaults;

/* ===== CURRENT FLOOR ===== */
function getCurrentFloorData() {
  return FLOOR_DATA[currentFloor] ?? { cameras:[] };
}

/* ===== RENDER FLOOR PLAN (inline SVG) ===== */
function renderFloorPlan() {
  const canvas = document.getElementById("floorCanvas");
  if (!canvas) return;
  const plan = FLOOR_PLANS[currentFloor];
  if (!plan) return;
  const floorData = getCurrentFloorData();

  const camSVGs = floorData.cameras.map(cam => {
    const status = getStatus(cam.trust);
    const borderColor = status==="red" ? "#ff4d4d" : status==="yellow" ? "#f5c518" : "#3ddc84";
    const dotAnim     = status==="red" ? ' class="blink-dot"' : '';
    const isSelected  = selectedCamera?.id === cam.id;
    const selExtra    = isSelected
      ? `<rect x="${cam.cx-cam.w/2-4}" y="${cam.cy-cam.h/2-4}" width="${cam.w+8}" height="${cam.h+8}" rx="12" fill="none" stroke="#4a7cff" stroke-width="3" opacity="0.7"/>`
      : '';
    const x = cam.cx - cam.w/2;
    const y = cam.cy - cam.h/2;
    const labelW = Math.max(cam.id.length * 7 + 14, 56);

    return `
      <g class="cam-group" data-id="${cam.id}" style="cursor:pointer">
        ${selExtra}
        <rect x="${x}" y="${y}" width="${cam.w}" height="${cam.h}"
              rx="9" fill="rgba(28,32,40,0.82)" stroke="${borderColor}" stroke-width="${isSelected?4:3}"/>

        <!-- camera icon (centred top area) -->
        <g transform="translate(${cam.cx-13},${y+10})">
          <path d="M22 6l-7 5 7 5V6z" fill="none" stroke="rgba(255,255,255,0.65)" stroke-width="1.8"/>
          <rect x="1" y="4" width="14" height="13" rx="2" fill="none" stroke="rgba(255,255,255,0.65)" stroke-width="1.8"/>
        </g>

        <!-- status dot (top right) -->
        <circle cx="${x+cam.w-12}" cy="${y+14}" r="6" fill="${borderColor}"${dotAnim}/>

        <!-- camera ID label (above box) -->
        <rect x="${x}" y="${y-23}" width="${labelW}" height="20" rx="10" fill="rgba(30,32,40,0.78)"/>
        <text x="${x+8}" y="${y-8}" font-size="11" fill="#fff"
              font-family="'JetBrains Mono',monospace" font-weight="700">${cam.id}</text>

        <!-- place name (middle) -->
        <text x="${cam.cx}" y="${cam.cy+6}" text-anchor="middle" font-size="12"
              fill="rgba(255,255,255,0.72)" font-family="'Noto Sans KR',sans-serif" font-weight="600">${cam.place}</text>

        <!-- bottom badges -->
        <!-- people -->
        <rect x="${x+6}" y="${y+cam.h-27}" width="${Math.floor(cam.w/2)-10}" height="20" rx="4" fill="rgba(255,255,255,0.11)"/>
        <path d="M${x+14} ${y+cam.h-13} a4 4 0 0 1 8 0" fill="none" stroke="rgba(255,255,255,0.75)" stroke-width="1.5"/>
        <circle cx="${x+18}" cy="${y+cam.h-20}" r="3" fill="none" stroke="rgba(255,255,255,0.75)" stroke-width="1.5"/>
        <text x="${x+28}" y="${y+cam.h-11}" font-size="11" fill="rgba(255,255,255,0.9)"
              font-family="'JetBrains Mono',monospace" font-weight="700">${cam.people}</text>
        <!-- trust -->
        <rect x="${cam.cx+4}" y="${y+cam.h-27}" width="${Math.floor(cam.w/2)-10}" height="20" rx="4" fill="rgba(255,255,255,0.11)"/>
        <text x="${cam.cx + Math.floor(cam.w/4) - 2}" y="${y+cam.h-11}" font-size="11" fill="rgba(255,255,255,0.9)"
              font-family="'JetBrains Mono',monospace" font-weight="700" text-anchor="middle">${cam.trust}%</text>
      </g>`;
  }).join("");

  canvas.innerHTML = `
    <svg viewBox="${plan.viewBox}" xmlns="http://www.w3.org/2000/svg"
         width="100%" height="100%" style="display:block;">
      <defs>
        <style>
          @keyframes blinkAnim{0%,100%{opacity:1}50%{opacity:.25}}
          .blink-dot{animation:blinkAnim 1s infinite}
          .cam-group rect:first-of-type{transition:opacity .15s}
          .cam-group:hover rect:first-of-type{opacity:1!important}
        </style>
      </defs>
      ${plan.svgDefs}
      ${camSVGs}
    </svg>`;

  canvas.querySelectorAll(".cam-group").forEach(g => {
    g.addEventListener("click", () => {
      const camId = g.dataset.id;
      const fd = getCurrentFloorData();
      selectedCamera = fd.cameras.find(c => c.id === camId) ?? null;
      renderFloorPlan();
      updatePanels();
      openCameraPreviewPopup(camId);
    });
  });
}

/* ===== RENDER TABS ===== */
function renderFloorTabs() {
  const wrap = document.getElementById("floorTabs");
  wrap.innerHTML = "";
  Object.keys(FLOOR_DATA).map(Number).sort((a,b)=>a-b).forEach(floor => {
    const btn = document.createElement("button");
    btn.className = `floor-tab${floor===currentFloor?" active":""}`;
    btn.textContent = `${floor}층`;
    btn.addEventListener("click", () => { currentFloor=floor; initialCameraId=null; renderAll(); });
    wrap.appendChild(btn);
  });
}

/* ===== RENDER CAMERA LIST ===== */
function renderCameraList() {
  const listEl = document.getElementById("cameraList");
  const floor  = getCurrentFloorData();
  listEl.innerHTML = "";
  floor.cameras.forEach(cam => {
    const status = getStatus(cam.trust);
    const item   = document.createElement("div");
    item.className = `camera-list-item${selectedCamera?.id===cam.id?" selected":""}`;
    item.innerHTML = `
      <span class="camera-list-dot ${status}"></span>
      <span class="camera-list-id">${cam.id}</span>
      <span class="camera-list-info">${cam.people}명 · ${cam.trust}%</span>`;
    item.addEventListener("click", () => { selectedCamera=cam; renderFloorPlan(); updatePanels(); openCameraPreviewPopup(cam.id); });
    listEl.appendChild(item);
  });
}

/* ===== UPDATE PANELS ===== */
function updatePanels() {
  const floor = getCurrentFloorData();
  if (!selectedCamera) selectedCamera = floor.cameras[0] ?? null;
  const cam = selectedCamera;
  if (!cam) return;
  const status = getStatus(cam.trust);
  const label  = statusText(status);

  document.getElementById("selectedInlineId").textContent    = cam.id;
  document.getElementById("selectedInlineMeta").textContent  =
    `${currentFloor}층 · ${cam.place} · ${cam.people}명 · ${cam.trust}%`;
  const inlineBadge = document.getElementById("selectedInlineStatus");
  inlineBadge.textContent = label;
  inlineBadge.className   = `selected-inline-status ${status}`;

  document.getElementById("detail-id").textContent    = cam.id;
  document.getElementById("detail-place").textContent = cam.place ?? `${currentFloor}층`;
  document.getElementById("detail-people").textContent = `${cam.people}명`;
  document.getElementById("detail-trust").textContent  = `${cam.trust}%`;
  const badge = document.getElementById("detail-status");
  badge.textContent = label;
  badge.className   = `detail-status-badge ${status}`;

  renderCameraList();
}

/* ===== UPDATE SUMMARY ===== */
function updateSummary() {
  const floor      = getCurrentFloorData();
  const alertCams  = floor.cameras.filter(c => getStatus(c.trust)!=="green");
  const totalPeople = floor.cameras.reduce((s,c)=>s+c.people,0);
  const alertPeople = alertCams.reduce((s,c)=>s+c.people,0);

  document.getElementById("summary-floor").textContent    = `${currentFloor}층`;
  document.getElementById("summary-people").textContent   = `${totalPeople}명`;
  document.getElementById("summary-cameras").textContent  = `${floor.cameras.length}대`;
  document.getElementById("summary-alerts").textContent   = `${alertCams.length}대`;
  document.getElementById("summary-alert-people").textContent = `${alertPeople}명`;
  document.getElementById("layout-title").textContent     = `${currentFloor}층 도면`;
  document.getElementById("floor-total-badge").textContent = `${totalPeople}명`;

  const chip = document.getElementById("floorStatusChip");
  if (chip) {
    const hasRed = floor.cameras.some(c=>getStatus(c.trust)==="red");
    if (alertCams.length===0) { chip.textContent="정상"; chip.className="status-chip safe"; }
    else { chip.textContent=`주의 필요 ${alertCams.length}`; chip.className=`status-chip ${hasRed?"danger":"warning"}`; }
  }
}





/* ===== CCTV PREVIEW POPUP ===== */
function openCameraPreviewPopup(camId) {
  const camera = findCameraById(camId) || selectedCamera;
  if (!camera) return;

  selectedCamera = camera;
  updatePanels();

  const overlay = document.getElementById("camera-preview-overlay");
  const screen = document.getElementById("camera-preview-screen");
  const subtitle = document.getElementById("camera-preview-subtitle");
  const tag = document.getElementById("camera-preview-tag");
  const alertBox = document.getElementById("camera-preview-alert");
  const confidence = document.getElementById("camera-preview-confidence");
  const place = document.getElementById("camera-preview-place");
  const people = document.getElementById("camera-preview-people");
  const trust = document.getElementById("camera-preview-trust");
  const statusEl = document.getElementById("camera-preview-status");

  const status = getStatus(camera.trust);
  const label = statusText(status);

  subtitle.textContent = `${camera.floor}층 · ${camera.place} · ${camera.id}`;
  tag.textContent = `${camera.id} · ${camera.floor}층 CCTV`;
  confidence.textContent = `${camera.trust}%`;
  place.textContent = `${camera.floor}층 ${camera.place}`;
  people.textContent = `${camera.people}명`;
  trust.textContent = `${camera.trust}%`;
  statusEl.textContent = label;
  statusEl.className = `status-${status}`;

  screen.className = `camera-preview-screen status-${status}`;
  if (status === 'red') {
    alertBox.classList.remove('hidden');
  } else {
    alertBox.classList.add('hidden');
  }

  overlay.classList.remove('hidden');
}

function openSelectedCameraPopup() {
  if (!selectedCamera) return;
  openCameraPreviewPopup(selectedCamera.id);
}

function closeCameraPreviewPopup() {
  const overlay = document.getElementById("camera-preview-overlay");
  if (!overlay) return;
  overlay.classList.add('hidden');
}

/* ===== FLOOR NOTIFICATIONS ===== */
let currentNotifCamId = null;

function getAllAlertCameras() {
  return Object.entries(FLOOR_DATA)
    .flatMap(([floorNum, floorData]) =>
      floorData.cameras.map(camera => ({
        ...camera,
        floor: Number(floorNum),
        status: getStatus(camera.trust),
      }))
    )
    .filter(camera => camera.status !== "green")
    .sort((a, b) => {
      const order = { red: 0, yellow: 1, green: 2 };
      return order[a.status] - order[b.status] || a.trust - b.trust;
    });
}

function buildFloorNotifications() {
  const alerts = getAllAlertCameras();

  const listEl = document.getElementById("notif-list-items");
  const emptyEl = document.getElementById("notif-list-empty");
  const bellDot = document.getElementById("bell-dot");

  if (!listEl || !emptyEl || !bellDot) return;

  listEl.innerHTML = "";

  if (alerts.length === 0) {
    emptyEl.classList.remove("hidden");
    bellDot.classList.remove("active");
    return;
  }

  emptyEl.classList.add("hidden");
  bellDot.classList.add("active");

  alerts.forEach(alert => {
    const row = document.createElement("div");
    row.className = "notif-item";

    row.innerHTML = `
      <span class="notif-dot ${alert.status}"></span>

      <div class="notif-item-text">
        <div class="notif-item-title">${alert.id} · ${alert.floor}층</div>
        <div class="notif-item-sub">
          ${alert.place} · 신뢰도 ${alert.trust}% · 인원 ${alert.people}명 · ${statusText(alert.status)}
        </div>
      </div>

      <span class="notif-item-action">확인 →</span>
    `;

    row.addEventListener("click", () => {
      closeFloorNotifList();
      openFloorNotifPopup(alert.id);
    });

    listEl.appendChild(row);
  });
}

function toggleFloorNotifList() {
  const list = document.getElementById("notif-list");
  if (!list) return;
  list.classList.toggle("hidden");
}

function closeFloorNotifList() {
  const list = document.getElementById("notif-list");
  if (!list) return;
  list.classList.add("hidden");
}

function findCameraById(camId) {
  for (const [floorNum, floorData] of Object.entries(FLOOR_DATA)) {
    const camera = floorData.cameras.find(item => item.id === camId);
    if (camera) {
      return {
        ...camera,
        floor: Number(floorNum),
        status: getStatus(camera.trust),
      };
    }
  }

  return null;
}

function openFloorNotifPopup(camId) {
  const camera = findCameraById(camId);
  if (!camera) return;

  currentNotifCamId = camId;

  const overlay = document.getElementById("notif-overlay");
  const body = document.getElementById("notif-popup-body");
  if (!overlay || !body) return;

  const previewHtml =
    camera.status === "red"
      ? `
        <div class="popup-camera-preview">
          <div class="popup-camera-tag">${camera.id} · ${camera.floor}층 CCTV</div>

          <span class="popup-camera-icon">
            <svg
              width="44"
              height="44"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="1.5"
            >
              <path d="M23 7l-7 5 7 5V7z"></path>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
          </span>

          <div class="popup-camera-confidence">${camera.trust}%</div>
        </div>
      `
      : "";

  const message =
    camera.status === "red"
      ? `카메라 신뢰도가 <b style="color:#ff4d4d">${camera.trust}%</b>로 매우 낮습니다. 즉시 점검이 필요합니다.`
      : `카메라 신뢰도가 <b style="color:#f5c518">${camera.trust}%</b>로 주의 수준입니다. 확인이 필요합니다.`;

  body.innerHTML = `
    ${previewHtml}

    <div class="popup-detail">
      <strong>${camera.id} — ${camera.floor}층 ${camera.place}</strong>
      ${message}

      <div class="popup-status ${camera.status}">
        ${statusText(camera.status)} · 인원 ${camera.people}명
      </div>
    </div>
  `;

  overlay.classList.remove("hidden");
}

function closeFloorNotifPopup() {
  const overlay = document.getElementById("notif-overlay");
  if (!overlay) return;

  overlay.classList.add("hidden");
  currentNotifCamId = null;
}

function goToFloorCamera() {
  const camera = findCameraById(currentNotifCamId);
  if (!camera) return;

  const params = new URLSearchParams({ floor: String(camera.floor), cam: camera.id });
  window.location.href = "floordashboard.html?" + params.toString();
}

document.addEventListener("click", event => {
  const bell = document.getElementById("notif-bell");
  const list = document.getElementById("notif-list");

  if (!bell || !list) return;

  if (!bell.contains(event.target) && !list.contains(event.target)) {
    closeFloorNotifList();
  }
});


/* ===== CLOCK ===== */
function updateClock() {
  const now = new Date();
  document.getElementById("sidebar-date").textContent =
    `${now.getFullYear()}년 ${now.getMonth()+1}월 ${now.getDate()}일`;
  document.getElementById("sidebar-time").textContent =
    now.toLocaleTimeString("ko-KR",{hour:"2-digit",minute:"2-digit"});
}

/* ===== URL PARAMS ===== */
function applyUrlParams() {
  const p = new URLSearchParams(window.location.search);
  const fl = Number(p.get("floor"));
  const cam = p.get("cam");
  const openMode = p.get("open");

  const floorFromCam = Number(String(cam || "").match(/^(\d+)F/)?.[1]);

  if (FLOOR_DATA[fl]) {
    currentFloor = fl;
  } else if (FLOOR_DATA[floorFromCam]) {
    currentFloor = floorFromCam;
  }

  if (cam) initialCameraId = cam;
  shouldOpenInitialCamera = openMode === "cctv" || openMode === "1" || openMode === "true";
}

/* ===== RENDER ALL ===== */
function renderAll() {
  const floor = getCurrentFloorData();
  renderFloorTabs();
  updateSummary();
  selectedCamera =
    floor.cameras.find(c=>c.id===initialCameraId) ||
    floor.cameras[0] || null;
  renderFloorPlan();
  updatePanels();
  buildFloorNotifications();

  if (shouldOpenInitialCamera && selectedCamera) {
    const camIdToOpen = selectedCamera.id;
    shouldOpenInitialCamera = false;
    setTimeout(() => openCameraPreviewPopup(camIdToOpen), 0);
  }
}

/* ===== REFRESH ===== */
function refresh() { buildFloorData(); applyUrlParams(); renderAll(); }

window.addEventListener("storage", e => {
  if ([AI_RESULTS_KEY, BUILDING_CONFIG_KEY, CAMERA_LAYOUT_KEY, "adminCameras", "adminAlertThresholds"].includes(e.key)) refresh();
});

window.addEventListener("DOMContentLoaded", () => {
  updateClock();
  setInterval(updateClock, 1000);
  buildFloorData();
  applyUrlParams();
  renderAll();
});

document.addEventListener("click", event => {
  const overlay = document.getElementById("camera-preview-overlay");
  if (!overlay || overlay.classList.contains("hidden")) return;
  if (event.target === overlay) closeCameraPreviewPopup();
});

document.addEventListener("keydown", event => {
  if (event.key === "Escape") {
    closeCameraPreviewPopup();
    closeFloorNotifPopup();
  }
});
