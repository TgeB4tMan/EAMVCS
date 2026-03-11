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
const BACKEND_URL = "http://127.0.0.1:8000";

// ===========================
// CORE STATE RESET
// ===========================
function resetVoiceState() {
    recordedAudio = null;
    audioChunks = [];
    isRecording = false;
    
    // Reset UI elements
    const fileInput = document.getElementById("audioFile");
    if (fileInput) fileInput.value = "";
    
    const status = document.getElementById("recordingStatus");
    if (status) status.innerText = "Click to start recording";
    
    const recordBtn = document.getElementById("recordBtn");
    if (recordBtn) recordBtn.classList.remove("recording");
    
    // Reset preview players
    if (window.referenceAudio) {
        window.referenceAudio.pause();
        window.referenceAudio = null;
    }
    
    const refTimeline = document.getElementById('refTimeline');
    if (refTimeline) refTimeline.style.width = '0%';
    
    const refDuration = document.getElementById('refDuration');
    if (refDuration) refDuration.innerText = '0:00';
}



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
        const charCount = document.getElementById("charCount");
        if (charCount) charCount.innerText = this.value.length;
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
                
                // FIX: If we just recorded, we must ignore any old uploaded file
                const fileInput = document.getElementById("audioFile");
                if (fileInput) fileInput.value = ""; 
                
                showNotification("Recording saved", "success");

            };

            mediaRecorder.start();

            isRecording = true;

            const recordBtn = document.getElementById("recordBtn");
            if (recordBtn) recordBtn.classList.add("recording");

            const recordingStatus = document.getElementById("recordingStatus");
            if (recordingStatus) recordingStatus.innerText = "Recording...";

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

        const recordBtn = document.getElementById("recordBtn");
        if (recordBtn) recordBtn.classList.remove("recording");

        const recordingStatus = document.getElementById("recordingStatus");
        if (recordingStatus) recordingStatus.innerText = "Recording stopped";

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
        // Double check if file input has something
        const fileInput = document.getElementById("audioFile");
        if (fileInput && fileInput.files.length > 0) {
            recordedAudio = fileInput.files[0];
        } else {
            showNotification("Please upload or record voice first", "warning");
            return;
        }
    }
    navigateTo("generate");
    setupReferencePlayer(); // Refresh player with current audio
}

// ===========================
// GENERATE VOICE (REAL EMOTION AI)
// ===========================

// MP3 to WAV conversion using Web Audio API
async function convertMp3ToWav(mp3Blob) {
    return new Promise((resolve) => {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const reader = new FileReader();
        
        reader.onload = async () => {
            const arrayBuffer = reader.result;
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // Convert to WAV
            const length = audioBuffer.length;
            const sampleRate = audioBuffer.sampleRate;
            const numberOfChannels = audioBuffer.numberOfChannels;
            const buffer = new ArrayBuffer(44 + length * 2);
            const view = new DataView(buffer);
            
            // WAV header
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numberOfChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * numberOfChannels * 2, true);
            view.setUint16(32, numberOfChannels * 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, length * 2, true);
            
            // Convert float samples to 16-bit PCM
            const volume = 0.8;
            let offset = 44;
            for (let i = 0; i < length; i++) {
                let sample = audioBuffer.getChannelData(0)[i] * volume;
                sample = Math.max(-1, Math.min(1, sample));
                view.setInt16(offset, sample * 0x7FFF, true);
                offset += 2;
            }
            
            resolve(new Blob([buffer], { type: 'audio/wav' }));
        };
        
        reader.readAsArrayBuffer(mp3Blob);
    });
}

async function generateVoice() {
    const text = document.getElementById("textInput").value;
    const language = document.getElementById("languageSelect").value;
    const alpha = parseFloat(document.getElementById("emotionSlider").value) / 100;
    const fileInput = document.getElementById("audioFile");
    const uploadedAudio = fileInput.files[0];

    // FIX: Core State Management
    let audioBlob = null;
    if (uploadedAudio) {
        audioBlob = uploadedAudio;
        recordedAudio = null; // Clear old recording to avoid confusion
    } else if (recordedAudio) {
        audioBlob = recordedAudio;
    }

    if (text.trim() === "") {
        showNotification("Please enter some text to speak.", "warning");
        return;
    }

    if (!audioBlob) {
        showNotification("Please provide a reference voice (record or upload).", "warning");
        return;
    }

    showLoading(true);
    const start = Date.now();

    try {
        const formData = new FormData();
        formData.append("text", text);
        formData.append("language", language);
        formData.append("alpha", alpha.toString());
        formData.append("audio", audioBlob, "voice_ref.wav");

        const response = await fetch(`${BACKEND_URL}/synthesize`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        // Load audio from the new endpoint
        const audioResponse = await fetch(`${BACKEND_URL}/audio/${result.audio_path}`);
        generatedAudio = await audioResponse.blob();
        
        // Extract real metrics from JSON response
        const voiceSimilarity = result.voice_similarity;
        const valence = result.valence;
        const arousal = result.arousal;
        const dominance = result.dominance;
        const emotionLabel = result.emotion_detected;
        const confidence = result.confidence;
        
        console.log(`Metrics Received - Emotion: ${emotionLabel} (${confidence}%)`);
        
        const time = ((Date.now() - start) / 1000).toFixed(1);
        document.getElementById("genTime").innerText = time + "s";
        document.getElementById("audioLang").innerText = 
            document.getElementById("languageSelect").selectedOptions[0].text;
        
        // Update metrics with real values
        if (voiceSimilarity) {
            const similarityPercent = (parseFloat(voiceSimilarity) * 100).toFixed(1) + "%";
            document.querySelector(".metrics-grid .metric:nth-child(1) .metric-value").innerText = similarityPercent;
        }
        
        if (emotionLabel && confidence) {
            const accDisplay = `${emotionLabel.toUpperCase()} (${confidence.toFixed(1)}%)`;
            document.querySelector(".metrics-grid .metric:nth-child(2) .metric-value").innerText = accDisplay;
        }
        
        // Update emotion visualization
        if (arousal) {
            const arousalValue = parseFloat(arousal);
            document.querySelector(".bg-animation").style.filter = 
                `hue-rotate(${arousalValue * 180}deg)`;
        }
        
        // Create emotion profile bars
        createEmotionBars(valence, arousal, dominance);
        
        setupRealAudioPlayer();
        navigateTo("results");
        drawWaveform();
        
        // Show Feedback Section
        const feedbackContainer = document.getElementById('feedbackContainer');
        if (feedbackContainer) feedbackContainer.style.display = 'block';
        
        const correctionSection = document.getElementById('correctionSection');
        if (correctionSection) correctionSection.style.display = 'none';

        showNotification("Voice generated successfully", "success");

    } catch (err) {
        showNotification("Error: " + err.message, "error");
        console.error(err);
    } finally {
        showLoading(false);
    }
}


// ===========================
// REAL AUDIO PLAYER SETUP
// ===========================

function setupRealAudioPlayer() {
    // Add HTML5 audio element if not exists
    if (!document.getElementById('resultAudio')) {
        const audioContainer = document.querySelector('.audio-result');
        const audioEl = document.createElement('audio');
        audioEl.id = 'resultAudio';
        audioEl.controls = true;
        audioEl.style.width = '100%';
        audioEl.style.marginTop = '20px';
        audioContainer.appendChild(audioEl);
    }
    
    // Set audio source
    const audioEl = document.getElementById('resultAudio');
    audioEl.src = URL.createObjectURL(generatedAudio);
    
    // Update duration display
    audioEl.addEventListener('loadedmetadata', () => {
        const duration = formatTime(audioEl.duration);
        document.getElementById('audioDuration').innerText = duration;
    });
    
    // Update play button
    audioEl.addEventListener('play', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    });
    
    audioEl.addEventListener('pause', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
    });
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// ===========================
// PLAY RESULT (Updated)
// ===========================

function playResult() {
    if (wavesurfer) {
        wavesurfer.playPause();
    } else {
        const audioEl = document.getElementById('resultAudio');
        if (audioEl) {
            if (audioEl.paused) audioEl.play();
            else audioEl.pause();
        }
    }
}

// ===========================
// REFERENCE AUDIO PLAYER (Real-time)
// ===========================

function setupReferencePlayer() {
    if (!recordedAudio) return;
    
    // Create real-time progress for reference audio
    const refAudio = new Audio(URL.createObjectURL(recordedAudio));
    
    refAudio.addEventListener('loadedmetadata', () => {
        const duration = formatTime(refAudio.duration);
        const durationEl = document.getElementById('refDuration');
        if (durationEl) durationEl.innerText = duration;
    });
    
    refAudio.addEventListener('timeupdate', () => {
        const percent = (refAudio.currentTime / refAudio.duration) * 100;
        const timeline = document.getElementById('refTimeline');
        if (timeline) timeline.style.width = percent + '%';
    });
    
    // Store reference for play function
    window.referenceAudio = refAudio;
}

function playReference() {
    if (window.referenceAudio) {
        if (window.referenceAudio.paused) {
            window.referenceAudio.play();
        } else {
            window.referenceAudio.pause();
        }
    } else if (recordedAudio) {
        setupReferencePlayer();
        window.referenceAudio.play();
    }
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
    resetVoiceState();
    navigateTo("upload");
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
// LOADING OVERLAY
// ===========================

function showLoading(show) {

    const overlay = document.getElementById("loadingOverlay");
    if (overlay) overlay.style.display = show ? "flex" : "none";

}


// ===========================
// REAL WAVEFORM (WaveSurfer.js)
// ===========================

let wavesurfer = null;

function drawWaveform() {
    if (!generatedAudio) return;

    // Destroy existing wavesurfer if it exists to prevent duplicate rendering
    if (wavesurfer) {
        wavesurfer.destroy();
    }

    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#4facfe',
        progressColor: '#00f2fe',
        cursorColor: '#ffffff',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 100,
        barGap: 3,
        interact: true,
        fillParent: true
    });
    
    // Load the generated audio
    wavesurfer.load(URL.createObjectURL(generatedAudio));
    
    // Sync UI play button icon
    wavesurfer.on('play', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    });
    
    wavesurfer.on('pause', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
    });

    wavesurfer.on('finish', () => {
        const playBtn = document.querySelector('.play-btn-large');
        if (playBtn) playBtn.innerHTML = '<i class="fas fa-play"></i>';
    });
}

// ===========================
// DYNAMIC EMOTION BARS
// ===========================

function createEmotionBars(valence, arousal, dominance) {
    // Add emotion bars to metrics section if not exists
    if (!document.querySelector('.emotion-bars')) {
        const metricsSection = document.querySelector('.quality-metrics');
        const emotionBars = document.createElement('div');
        emotionBars.className = 'emotion-bars';
        emotionBars.innerHTML = `
            <div class="emotion-header">
                <h4>VAD Emotion Intelligence Model</h4>
                <div class="info-icon">ⓘ
                    <div class="info-tooltip">
                        <p><strong>Valence:</strong> Measures how positive or negative the emotion is.</p>
                        <p><strong>Arousal:</strong> Measures the intensity and energy level.</p>
                        <p><strong>Dominance:</strong> Measures the degree of control expressed.</p>
                    </div>
                </div>
            </div>
            <div class="emotion-bar-container">
                <div class="emotion-bar-item">
                    <div class="bar-label">
                        <span>Valence <small>(Happiness vs Sadness)</small></span>
                        <span id="valenceValue" class="vad-number">0.00</span>
                    </div>
                    <div class="emotion-bar-track">
                        <div id="valenceBar" class="emotion-bar-fill"></div>
                    </div>
                </div>
                <div class="emotion-bar-item">
                    <div class="bar-label">
                        <span>Arousal <small>(Excitement vs Calm)</small></span>
                        <span id="arousalValue" class="vad-number">0.00</span>
                    </div>
                    <div class="emotion-bar-track">
                        <div id="arousalBar" class="emotion-bar-fill"></div>
                    </div>
                </div>
                <div class="emotion-bar-item">
                    <div class="bar-label">
                        <span>Dominance <small>(Confidence vs Fear)</small></span>
                        <span id="dominanceValue" class="vad-number">0.00</span>
                    </div>
                    <div class="emotion-bar-track">
                        <div id="dominanceBar" class="emotion-bar-fill"></div>
                    </div>
                </div>
            </div>
            <div class="vad-explanation-card">
                <p id="vadDescription">Detecting emotional nuances...</p>
            </div>
        `;
        metricsSection.appendChild(emotionBars);
    }
    
    // Update emotion bars with real values
    if (valence !== null && arousal !== null && dominance !== null) {
        const vBar = document.getElementById('valenceBar');
        if (vBar) vBar.style.width = (valence * 100) + '%';
        const vVal = document.getElementById('valenceValue');
        if (vVal) vVal.innerText = valence.toFixed(2);
        
        const aBar = document.getElementById('arousalBar');
        if (aBar) aBar.style.width = (arousal * 100) + '%';
        const aVal = document.getElementById('arousalValue');
        if (aVal) aVal.innerText = arousal.toFixed(2);
        
        const dBar = document.getElementById('dominanceBar');
        if (dBar) dBar.style.width = (dominance * 100) + '%';
        const dVal = document.getElementById('dominanceValue');
        if (dVal) dVal.innerText = dominance.toFixed(2);

        // Dynamic Text Explanation
        const descEl = document.getElementById('vadDescription');
        if (descEl) {
            let desc = "";
            // More sensitive thresholds for a 'High Fidelity' feel
            if (valence > 0.65) desc += "The voice sounds **Positive and Happy**. ";
            else if (valence < 0.35) desc += "The voice sounds **Negative or Sad**. ";
            else if (valence > 0.52) desc += "The tone is **Pleasant and Bright**. ";
            else if (valence < 0.48) desc += "The tone is **Serious and Somber**. ";
            else desc += "The emotional tone is **Balanced and Neutral**. ";

            if (arousal > 0.65) desc += "There is **High Energy** and excitement. ";
            else if (arousal < 0.35) desc += "The recording is **Calm and Muted**. ";
            else if (arousal > 0.55) desc += "The energy level is **Active**. ";
            else if (arousal < 0.45) desc += "The energy level is **Relaxed**. ";

            descEl.innerHTML = desc;
        }
    }
}

// ===========================
// FEEDBACK SYSTEM
// ===========================

function showCorrection() {
    const section = document.getElementById('correctionSection');
    if (section) section.style.display = 'block';
}

async function submitFeedback(isCorrect) {
    const feedbackContainer = document.getElementById('feedbackContainer');
    // Extract predicted emotion from the UI
    const metricsSection = document.querySelector(".metrics-grid .metric:nth-child(2) .metric-value").innerText;
    const predictedEmotion = metricsSection.split(' ')[0].toLowerCase();
    
    let correctEmotion = predictedEmotion;

    if (!isCorrect) {
        correctEmotion = document.getElementById('correctEmotionSelect').value;
    }

    try {
        const formData = new FormData();
        const audioToUpload = recordedAudio || document.getElementById("audioFile").files[0];
        formData.append("audio", audioToUpload, "feedback.wav");
        formData.append("correct_emotion", correctEmotion);
        formData.append("predicted_emotion", predictedEmotion);

        const response = await fetch(`${BACKEND_URL}/feedback`, {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            showNotification("Feedback received! Thank you.", "success");
            
            // Temporary success message without destroying the structure
            const originalContent = feedbackContainer.innerHTML;
            feedbackContainer.innerHTML = "<h4>Feedback saved successfully!</h4>";
            
            setTimeout(() => {
                feedbackContainer.style.display = 'none';
                feedbackContainer.innerHTML = originalContent; // Restore it for next time
            }, 2000);
        } else {
            throw new Error("Failed to send feedback");
        }
    } catch (error) {
        showNotification("Error: " + error.message, "error");
    }
}

// ===========================
// NOTIFICATIONS
// ===========================

function showNotification(message, type = "info") {
    const div = document.createElement("div");
    div.className = `notification notification-${type}`;
    div.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">✖</button>
    `;

    document.body.appendChild(div);

    setTimeout(() => {
        div.style.opacity = '0';
        div.style.transform = 'translateX(20px)';
        setTimeout(() => div.remove(), 300);
    }, 4000);
}
// ===========================
// ALPHA SLIDER DISPLAY
// ===========================
const emotionSlider = document.getElementById("emotionSlider");
if (emotionSlider) {
    emotionSlider.addEventListener("input", function() {
        const val = (this.value / 100).toFixed(1);
        const display = document.getElementById("alphaVal");
        if (display) display.innerText = val;
    });
}

// ===========================
// TRAINING STATUS POLLING
// ===========================
async function checkTrainingStatus() {
    try {
        const response = await fetch(`${BACKEND_URL}/training-status`);
        const data = await response.json();
        
        const badge = document.getElementById("trainingStatus");
        if (badge) {
            badge.style.display = data.is_training ? "flex" : "none";
        }
        
        // Lock feedback buttons if training
        const feedbackButtons = document.getElementById("feedbackButtons");
        const trainingLockedMsg = document.getElementById("trainingLockedMsg");
        const correctionSection = document.getElementById("correctionSection");
        
        if (feedbackButtons && trainingLockedMsg) {
            if (data.is_training) {
                feedbackButtons.style.display = "none";
                if (correctionSection) correctionSection.style.display = "none";
                trainingLockedMsg.style.display = "block";
            } else {
                // Only show buttons if the locked message is currently being shown
                // (prevents overriding the 'Thank you' success message)
                if (trainingLockedMsg.style.display === "block") {
                    feedbackButtons.style.display = "flex";
                    trainingLockedMsg.style.display = "none";
                }
            }
        }
        
    } catch (e) {
        // Silently fail if backend is restarting
    }
}

// Poll every 15 seconds (reduce terminal log spam)
setInterval(checkTrainingStatus, 15000);
checkTrainingStatus(); // Initial check
