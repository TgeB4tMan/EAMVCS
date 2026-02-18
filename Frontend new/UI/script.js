// ===========================
// NeuroVoice script.js
// Frontend Demo Logic
// ===========================

// GLOBAL VARIABLES
let currentPage = "home";
let mediaRecorder;
let audioChunks = [];
let recordedAudio = null;
let isRecording = false;
let generatedAudio = null;


// ===========================
// PAGE NAVIGATION
// ===========================
function navigateTo(page) {

    document.querySelectorAll(".page").forEach(p => {
        p.classList.remove("active");
    });

    document.getElementById(page + "-page").classList.add("active");

    document.querySelectorAll(".nav-link").forEach(link => {
        link.classList.remove("active");
    });

    document.querySelectorAll(`.nav-link[data-page="${page}"]`).forEach(link => {
        link.classList.add("active");
    });

    currentPage = page;
}


// Navbar click support
document.querySelectorAll(".nav-link").forEach(link => {

    link.addEventListener("click", function (e) {

        e.preventDefault();

        const page = this.getAttribute("data-page");

        navigateTo(page);

    });

});


// ===========================
// TEXT CHARACTER COUNT
// ===========================

const textInput = document.getElementById("textInput");

if (textInput) {

    textInput.addEventListener("input", function () {

        document.getElementById("charCount").innerText = this.value.length;

    });

}


// ===========================
// RECORD VOICE
// ===========================

async function toggleRecording() {

    if (!isRecording) {

        try {

            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            mediaRecorder = new MediaRecorder(stream);

            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = () => {

                recordedAudio = new Blob(audioChunks, { type: "audio/webm" });

                showNotification("Recording saved", "success");

            };

            mediaRecorder.start();

            isRecording = true;

            document.getElementById("recordBtn").classList.add("recording");

            document.getElementById("recordingStatus").innerText = "Recording...";

        }

        catch {

            showNotification("Microphone access denied", "error");

        }

    }

    else {

        stopRecording();

    }

}


function stopRecording() {

    if (mediaRecorder && isRecording) {

        mediaRecorder.stop();

        isRecording = false;

        document.getElementById("recordBtn").classList.remove("recording");

        document.getElementById("recordingStatus").innerText = "Recording stopped";

    }

}


function playRecording() {

    if (!recordedAudio) {

        showNotification("No recording found", "warning");

        return;

    }

    const audio = new Audio(URL.createObjectURL(recordedAudio));

    audio.play();

}


// ===========================
// FILE UPLOAD
// ===========================

const audioFileInput = document.getElementById("audioFile");

if (audioFileInput) {

    audioFileInput.addEventListener("change", function () {

        if (this.files.length > 0) {

            recordedAudio = this.files[0];

            showNotification("Audio file loaded", "success");

        }

    });

}


// ===========================
// CONTINUE TO GENERATE
// ===========================

function proceedToGenerate() {

    if (!recordedAudio) {

        showNotification("Please upload or record voice first", "warning");

        return;

    }

    navigateTo("generate");

}


// ===========================
// GENERATE VOICE (FAKE AI)
// ===========================

function generateVoice() {

    const text = document.getElementById("textInput").value;

    if (text.trim() === "") {

        showNotification("Enter text first", "warning");

        return;

    }

    showLoading(true);

    const start = Date.now();

    setTimeout(() => {

        showLoading(false);

        const time = ((Date.now() - start) / 1000).toFixed(1);

        document.getElementById("genTime").innerText = time + "s";

        generatedAudio = recordedAudio;

        document.getElementById("audioLang").innerText =
            document.getElementById("languageSelect").selectedOptions[0].text;

        drawWaveform();

        navigateTo("results");

        showNotification("Voice generated successfully", "success");

    }, 2500);

}


// ===========================
// PLAY RESULT
// ===========================

function playResult() {

    if (!generatedAudio) return;

    const audio = new Audio(URL.createObjectURL(generatedAudio));

    audio.play();

}


// ===========================
// DOWNLOAD AUDIO
// ===========================

function downloadAudio() {

    if (!generatedAudio) return;

    const link = document.createElement("a");

    link.href = URL.createObjectURL(generatedAudio);

    link.download = "neurovoice.wav";

    link.click();

}


// ===========================
// GENERATE ANOTHER
// ===========================

function generateAnother() {

    navigateTo("generate");

}


// ===========================
// DEMO BUTTON
// ===========================

function showDemo() {

    showNotification("Demo coming soon!", "info");

}


// ===========================
// SHARE
// ===========================

function shareResult() {

    showNotification("Share feature coming soon", "info");

}


// ===========================
// REFERENCE PREVIEW
// ===========================

function playReference() {

    playRecording();

}


// ===========================
// LOADING OVERLAY
// ===========================

function showLoading(show) {

    const overlay = document.getElementById("loadingOverlay");

    overlay.style.display = show ? "flex" : "none";

}


// ===========================
// WAVEFORM DRAW
// ===========================

function drawWaveform() {

    const canvas = document.getElementById("waveform");

    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.beginPath();

    let x = 0;

    for (let i = 0; i < canvas.width; i += 10) {

        const height = Math.random() * 80;

        ctx.moveTo(x, 60 - height / 2);

        ctx.lineTo(x, 60 + height / 2);

        x += 10;

    }

    ctx.strokeStyle = "#4facfe";

    ctx.stroke();

}


// ===========================
// NOTIFICATIONS
// ===========================

function showNotification(message, type = "info") {

    const div = document.createElement("div");

    div.className = `notification notification-${type}`;

    div.innerHTML = `
        ${message}
        <button onclick="this.parentElement.remove()">✖</button>
    `;

    document.body.appendChild(div);

    setTimeout(() => {

        div.remove();

    }, 4000);

}
