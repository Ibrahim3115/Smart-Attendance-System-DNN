// Browser-only frontend using face-api.js
// Loads models from a public CDN and stores descriptors & attendance in localStorage

const MODEL_URI = "https://justadudewhohacks.github.io/face-api.js/models";
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const registerBtn = document.getElementById("registerBtn");
const matchBtn = document.getElementById("matchBtn");
const clearBtn = document.getElementById("clearBtn");
const nameInput = document.getElementById("nameInput");
const registeredList = document.getElementById("registeredList");
const statusEl = document.getElementById("status");
const downloadAttendance = document.getElementById("downloadAttendance");
const clearAttendance = document.getElementById("clearAttendance");

let canvas, displaySize;

async function init() {
  status("Loading models...");
  await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URI);
  await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URI);
  await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URI);
  status("Models loaded");

  // Enable controls now that models are ready
  [
    registerBtn,
    matchBtn,
    clearBtn,
    downloadAttendance,
    clearAttendance,
  ].forEach((b) => (b.disabled = false));
  nameInput.disabled = false;

  await startVideo();
  setupListeners();
  refreshRegisteredList();
}

function status(msg, type = "info") {
  const icon =
    type === "success"
      ? "✅"
      : type === "warn"
        ? "⚠️"
        : type === "error"
          ? "❌"
          : "ℹ️";
  statusEl.innerHTML = `<span class="status-icon">${icon}</span><span class="status-text">${msg}</span>`;
  statusEl.className = "status " + type;
}

async function startVideo() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      setupCanvas();
      runLoop();
    };
  } catch (e) {
    status("Error accessing camera: " + e.message, "error");
  }
}

function setupCanvas() {
  canvas = faceapi.createCanvasFromMedia(video);
  overlay.replaceWith(canvas);
  overlay.id = "overlay";
  displaySize = { width: video.videoWidth, height: video.videoHeight };
  canvas.width = displaySize.width;
  canvas.height = displaySize.height;
}

async function runLoop() {
  const detections = await faceapi
    .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (detections.length > 0) {
    const resized = faceapi.resizeResults(detections, displaySize);
    resized.forEach((d) => faceapi.draw.drawDetections(canvas, [d]));
  }
  requestAnimationFrame(runLoop);
}

function getEmbeddings() {
  try {
    return JSON.parse(localStorage.getItem("embeddings") || "{}");
  } catch (e) {
    return {};
  }
}

function saveEmbeddings(obj) {
  localStorage.setItem("embeddings", JSON.stringify(obj));
}

function refreshRegisteredList() {
  const e = getEmbeddings();
  registeredList.innerHTML = "";
  Object.keys(e).forEach((name) => {
    const li = document.createElement("li");
    li.className = "registered-item";
    li.innerHTML = `<span class="name">${name}</span> <span class="badge">✔︎</span>`;
    registeredList.appendChild(li);
  });
}

async function captureDescriptor() {
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();
  if (!detection) {
    status("No face detected. Please position your face clearly.", "warn");
    return null;
  }
  return Array.from(detection.descriptor);
}

registerBtn.addEventListener("click", async () => {
  const name = nameInput.value && nameInput.value.trim();
  if (!name) {
    status("Please enter a name", "warn");
    return;
  }
  status("Capturing face...");
  const desc = await captureDescriptor();
  if (!desc) return;
  const e = getEmbeddings();
  e[name] = desc;
  saveEmbeddings(e);
  refreshRegisteredList();
  status("Registered: " + name, "success");
});

matchBtn.addEventListener("click", async () => {
  status("Scanning...");
  const desc = await captureDescriptor();
  if (!desc) return;
  const e = getEmbeddings();
  if (Object.keys(e).length === 0) {
    status("No registered faces", "warn");
    return;
  }

  let best = { name: null, dist: Infinity };
  for (const [name, stored] of Object.entries(e)) {
    const d = euclideanDistance(desc, stored);
    if (d < best.dist) {
      best = { name, dist: d };
    }
  }
  const threshold = 0.6;
  if (best.name && best.dist < threshold) {
    status(
      `Matched: ${best.name} (distance: ${best.dist.toFixed(3)})`,
      "success",
    );
    markAttendance(best.name);
  } else {
    status("No match found (too far).", "warn");
  }
});

function euclideanDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

clearBtn.addEventListener("click", () => {
  if (confirm("Clear all registered faces?")) {
    localStorage.removeItem("embeddings");
    refreshRegisteredList();
    status("Cleared", "info");
  }
});

function markAttendance(name) {
  const attendance = JSON.parse(localStorage.getItem("attendance") || "[]");
  const today = new Date().toISOString().slice(0, 10);
  const already = attendance.some((r) => r.Name === name && r.Date === today);
  if (already) {
    status("Attendance already marked for " + name + " today", "info");
    return;
  }
  attendance.push({
    Name: name,
    Date: today,
    Time: new Date().toTimeString().slice(0, 8),
  });
  localStorage.setItem("attendance", JSON.stringify(attendance));
  status("Attendance marked for " + name, "success");
}

downloadAttendance.addEventListener("click", () => {
  const attendance = JSON.parse(localStorage.getItem("attendance") || "[]");
  if (attendance.length === 0) {
    status("No attendance records", "info");
    return;
  }
  const csv = [["Name", "Date", "Time"].join(",")]
    .concat(attendance.map((r) => `${r.Name},${r.Date},${r.Time}`))
    .join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `attendance_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
});

clearAttendance.addEventListener("click", () => {
  if (confirm("Clear attendance?")) {
    localStorage.removeItem("attendance");
    status("Attendance cleared", "info");
  }
});

function setupListeners() {
  video.addEventListener("play", () => {
    displaySize = { width: video.videoWidth, height: video.videoHeight };
    canvas = faceapi.createCanvasFromMedia(video);
    // append canvas inside the white camera frame so it overlays correctly
    const frame = document.querySelector(".camera-frame");
    if (frame) frame.appendChild(canvas);
    else document.querySelector(".video-col").appendChild(canvas);
  });
}

init();