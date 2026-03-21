// ===========================
// NeuroVoice script.js
// Frontend Demo Logic
// ===========================

// GLOBAL VARIABLES
let currentPage = "home";
let mediaRecorder;
let recordingStream = null;
let audioChunks = [];
let recordedAudio = null;
let isRecording = false;
let generatedAudio = null;
let activePreviewAudio = null;
let activePreviewUrl = null;
let isPreviewPlaying = false;
let latestSynthesisResult = null;
let lastSynthesisReferenceAudio = null;
let lastSynthesisReferenceFilename = "reference.wav";
const BACKEND_URL = "http://localhost:8000";
const MAX_AUDIO_SIZE_BYTES = 10 * 1024 * 1024;
const MIN_REF_AUDIO_SEC = 3;
const MAX_REF_AUDIO_SEC = 30;
const MAX_TEXT_LENGTH = 500;
const ALLOWED_UPLOAD_EXTENSIONS = new Set([".wav", ".mp3", ".webm", ".m4a", ".ogg", ".flac", ".aac"]);
const EMOTION_LABELS = {
    neutral: "Neutral",
    happy: "Happy",
    sad: "Sad",
    angry: "Angry",
};

function getSupportedRecorderOptions() {
    if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") {
        return null;
    }

    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4;codecs=mp4a.40.2",
        "audio/mp4",
        "audio/ogg;codecs=opus",
        "audio/ogg",
    ];

    for (const mimeType of candidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
            return { mimeType };
        }
    }

    return null;
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
            recordingStream = stream;

            // Clear any uploaded file when starting recording
            const audioFileInput = document.getElementById("audioFileInput");
            if (audioFileInput) {
                audioFileInput.value = "";
            }

            const recorderOptions = getSupportedRecorderOptions();
            mediaRecorder = recorderOptions ? new MediaRecorder(stream, recorderOptions) : new MediaRecorder(stream);

            audioChunks = [];

            mediaRecorder.ondataavailable = event => {
                if (event.data && event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const chunkType = audioChunks.find((chunk) => chunk && chunk.type)?.type;
                const resolvedType = chunkType || mediaRecorder?.mimeType || "audio/webm";
                if (audioChunks.length === 0) {
                    showNotification("Recording was empty. Please try again.", "error");
                    return;
                }

                resetPreviewPlayback();
                recordedAudio = new Blob(audioChunks, { type: resolvedType });
                showNotification("Recording saved", "success");
                
                // Update UI to show recorded audio is active
                const recordingStatus = document.getElementById("recordingStatus");
                if (recordingStatus) {
                    recordingStatus.innerText = "Recording saved - Using recorded voice";
                }

                if (recordingStream) {
                    recordingStream.getTracks().forEach((track) => track.stop());
                    recordingStream = null;
                }
                mediaRecorder = null;
            };

            mediaRecorder.start();

            isRecording = true;

            document.getElementById("recordBtn").classList.add("recording");

            document.getElementById("recordingStatus").innerText = "Recording...";

        }

        catch {

            showNotification("Microphone access denied", "error");
            if (recordingStream) {
                recordingStream.getTracks().forEach((track) => track.stop());
                recordingStream = null;
            }

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
    else if (recordingStream) {
        recordingStream.getTracks().forEach((track) => track.stop());
        recordingStream = null;
    }

}


function playRecording() {
    const previewBtn = document.getElementById("previewRecordingBtn");
    let fallbackAttempted = false;

    if (!recordedAudio) {
        showNotification("No recording found", "warning");
        return;
    }

    if (isPreviewPlaying) {
        showNotification("Preview already playing", "info");
        return;
    }

    resetPreviewPlayback();
    activePreviewUrl = URL.createObjectURL(recordedAudio);
    activePreviewAudio = new Audio(activePreviewUrl);
    isPreviewPlaying = true;
    if (previewBtn) {
        previewBtn.disabled = true;
    }

    const releasePreviewLock = () => {
        if (activePreviewAudio) {
            activePreviewAudio.pause();
            activePreviewAudio.src = "";
        }
        if (activePreviewUrl) {
            URL.revokeObjectURL(activePreviewUrl);
        }
        activePreviewAudio = null;
        activePreviewUrl = null;
        isPreviewPlaying = false;
        if (previewBtn) {
            previewBtn.disabled = false;
        }
    };

    activePreviewAudio.onended = releasePreviewLock;
    activePreviewAudio.onerror = async () => {
        if (!fallbackAttempted && recordedAudio && !isWavAudio(recordedAudio)) {
            fallbackAttempted = true;
            try {
                const converted = await convertAudioToWav(recordedAudio);
                recordedAudio = converted;
                if (audioFileInput) {
                    audioFileInput.value = "";
                }
                releasePreviewLock();
                playRecording();
                return;
            } catch (conversionError) {
                console.warn("Preview fallback conversion failed:", conversionError);
            }
        }

        releasePreviewLock();
        showNotification("Preview failed. Try another audio file.", "error");
    };

    activePreviewAudio.play().catch(() => {
        releasePreviewLock();
        showNotification("Unable to play preview audio.", "error");
    });
}


// ===========================
// FILE UPLOAD
// ===========================

const audioFileInput = document.getElementById("audioFile");
const fileUploadArea = document.getElementById("fileUploadArea");

function resetPreviewPlayback() {
    if (activePreviewAudio) {
        activePreviewAudio.pause();
        activePreviewAudio.src = "";
    }
    if (activePreviewUrl) {
        URL.revokeObjectURL(activePreviewUrl);
    }
    activePreviewAudio = null;
    activePreviewUrl = null;
    isPreviewPlaying = false;

    const previewBtn = document.getElementById("previewRecordingBtn");
    if (previewBtn) {
        previewBtn.disabled = false;
    }
}

function getFileExtension(filename) {
    const lower = (filename || "").toLowerCase();
    const dotIndex = lower.lastIndexOf(".");
    return dotIndex >= 0 ? lower.slice(dotIndex) : "";
}

function handleReferenceAudioSelection(file) {
    if (!file) {
        return false;
    }

    const extension = getFileExtension(file.name);
    if (!ALLOWED_UPLOAD_EXTENSIONS.has(extension)) {
        showNotification("Unsupported audio format.", "error");
        return false;
    }

    if (file.size > MAX_AUDIO_SIZE_BYTES) {
        showNotification("Audio file exceeds 10MB limit.", "error");
        return false;
    }

    resetPreviewPlayback();
    recordedAudio = file;
    showNotification(`Audio file loaded: ${file.name}`, "success");
    
    // Update UI to show uploaded file is active
    const recordingStatus = document.getElementById("recordingStatus");
    if (recordingStatus) {
        recordingStatus.innerText = `Using uploaded file: ${file.name}`;
    }
    return true;
}

if (audioFileInput) {
    audioFileInput.addEventListener("change", function () {
        if (!this.files || this.files.length === 0) {
            return;
        }
        if (!handleReferenceAudioSelection(this.files[0])) {
            this.value = "";
        }
    });
}

if (fileUploadArea) {
    fileUploadArea.addEventListener("dragover", (event) => {
        event.preventDefault();
        fileUploadArea.classList.add("drag-over");
    });

    fileUploadArea.addEventListener("dragleave", () => {
        fileUploadArea.classList.remove("drag-over");
    });

    fileUploadArea.addEventListener("drop", (event) => {
        event.preventDefault();
        fileUploadArea.classList.remove("drag-over");

        const droppedFile = event.dataTransfer?.files?.[0];
        if (!droppedFile) {
            return;
        }

        const accepted = handleReferenceAudioSelection(droppedFile);
        if (!accepted) {
            return;
        }

        if (audioFileInput) {
            try {
                const transfer = new DataTransfer();
                transfer.items.add(droppedFile);
                audioFileInput.files = transfer.files;
            } catch {
                // Some browsers block programmatic assignment; recordedAudio fallback still works.
            }
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
// GENERATE VOICE (REAL EMOTION AI)
// ===========================

function isWavAudio(blob) {
    const mimeType = (blob?.type || "").toLowerCase();
    return mimeType.includes("wav") || mimeType.includes("wave");
}

async function convertAudioToWav(inputBlob) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    try {
        const arrayBuffer = await inputBlob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer.slice(0));
        const length = audioBuffer.length;
        const sampleRate = audioBuffer.sampleRate;
        const channelCount = audioBuffer.numberOfChannels;
        const mono = new Float32Array(length);

        for (let channel = 0; channel < channelCount; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < length; i++) {
                mono[i] += channelData[i];
            }
        }

        for (let i = 0; i < length; i++) {
            mono[i] /= channelCount;
        }

        const buffer = new ArrayBuffer(44 + length * 2);
        const view = new DataView(buffer);
        const writeString = (offset, value) => {
            for (let i = 0; i < value.length; i++) {
                view.setUint8(offset + i, value.charCodeAt(i));
            }
        };

        writeString(0, "RIFF");
        view.setUint32(4, 36 + length * 2, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, "data");
        view.setUint32(40, length * 2, true);

        let offset = 44;
        for (let i = 0; i < length; i++) {
            const sample = Math.max(-1, Math.min(1, mono[i]));
            view.setInt16(offset, sample * 0x7fff, true);
            offset += 2;
        }

        return new Blob([buffer], { type: "audio/wav" });
    } catch (err) {
        throw new Error(`Unable to decode uploaded audio for conversion: ${err.message}`);
    } finally {
        await audioContext.close();
    }
}

async function getAudioDurationSec(audioBlob) {
    const objectUrl = URL.createObjectURL(audioBlob);
    return new Promise((resolve, reject) => {
        const tempAudio = document.createElement("audio");
        tempAudio.preload = "metadata";
        tempAudio.onloadedmetadata = () => {
            const duration = tempAudio.duration;
            URL.revokeObjectURL(objectUrl);
            resolve(duration);
        };
        tempAudio.onerror = () => {
            URL.revokeObjectURL(objectUrl);
            reject(new Error("Unable to read audio metadata"));
        };
        tempAudio.src = objectUrl;
    });
}

async function generateVoice() {
    const text = document.getElementById("textInput").value;
    const targetLang = document.getElementById("languageSelect").value;
    const refLang = document.getElementById("refLangSelect").value;
    const textLangMode = document.getElementById("textLangSelect")?.value || "auto";
    const optionalRefText = document.getElementById("refTextInput").value.trim();
    const alpha = parseFloat(document.getElementById("emotionSlider").value) / 100;
    const uploadedAudio = audioFileInput && audioFileInput.files ? audioFileInput.files[0] : null;
    const cleanedText = text.trim().replace(/\s+/g, " ");

    if (cleanedText === "") {
        showNotification("Enter text first", "warning");
        return;
    }
    if (cleanedText.length > MAX_TEXT_LENGTH) {
        showNotification(`Text too long. Max ${MAX_TEXT_LENGTH} characters.`, "warning");
        return;
    }

    if (!uploadedAudio && !recordedAudio) {
        showNotification("Please upload or record audio", "warning");
        return;
    }

    showLoading(true);
    const start = Date.now();

    try {
        // Prioritize recorded audio over uploaded file
        let audioBlob = recordedAudio || uploadedAudio;
        let uploadFilename = audioBlob?.name || "reference_audio.webm";
        
        // Debug: Show which audio source is being used
        console.log("🎤 Audio source being used:", {
            hasRecordedAudio: !!recordedAudio,
            hasUploadedAudio: !!uploadedAudio,
            audioBlobName: audioBlob?.name,
            isUsingRecorded: !!recordedAudio
        });

        if (!isWavAudio(audioBlob)) {
            try {
                audioBlob = await convertAudioToWav(audioBlob);
                uploadFilename = uploadFilename.replace(/\.[^/.]+$/, "") || "reference_audio";
                uploadFilename += ".wav";
            } catch (conversionError) {
                console.warn("Audio conversion skipped, using original file:", conversionError);
            }
        }

        if (audioBlob.size > MAX_AUDIO_SIZE_BYTES) {
            throw new Error("Reference audio exceeds 10MB limit.");
        }
        const durationSec = await getAudioDurationSec(audioBlob);
        if (Number.isFinite(durationSec)) {
            if (durationSec < MIN_REF_AUDIO_SEC || durationSec > MAX_REF_AUDIO_SEC) {
                throw new Error(
                    `Reference audio must be between ${MIN_REF_AUDIO_SEC}s and ${MAX_REF_AUDIO_SEC}s. `
                    + `Received ${durationSec.toFixed(1)}s.`
                );
            }
        }

        lastSynthesisReferenceAudio = audioBlob;
        lastSynthesisReferenceFilename = uploadFilename;

        const formData = new FormData();
        formData.append("text", cleanedText);
        formData.append("target_lang", targetLang);
        formData.append("ref_lang", refLang);
        formData.append("text_lang", textLangMode);
        formData.append("alpha", alpha.toString());
        formData.append("audio", audioBlob, uploadFilename);
        if (optionalRefText) {
            formData.append("ref_text", optionalRefText);
        }

        const response = await fetch(`${BACKEND_URL}/synthesize-v2`, {
            method: "POST",
            body: formData
        });

        let result = null;
        try {
            result = await response.json();
            console.log("📡 Full API response:", result);
        } catch (parseError) {
            throw new Error(`Invalid response from server: ${parseError.message}`);
        }

        if (!response.ok || result.error) {
            if (result?.code === "TRANSLATION_UNAVAILABLE") {
                throw new Error(
                    "Offline translation model not available. "
                    + "Use Target Script mode or install a local translation model."
                );
            }
            throw new Error(result?.error || `HTTP error! status: ${response.status}`);
        }

        if (!result.audio_path) {
            throw new Error("Backend did not return generated audio path.");
        }

        const audioResponse = await fetch(`${BACKEND_URL}/audio/${encodeURIComponent(result.audio_path)}`);
        if (!audioResponse.ok) {
            throw new Error(`Failed to download generated audio: ${audioResponse.status}`);
        }

        generatedAudio = await audioResponse.blob();
        latestSynthesisResult = result;
        
        const time = ((Date.now() - start) / 1000).toFixed(1);
        document.getElementById("genTime").innerText = time + "s";
        document.getElementById("audioLang").innerText = 
            document.getElementById("languageSelect").selectedOptions[0].text;
        
        const voiceSimilarityMetric = document.getElementById("voiceSimilarityMetric");
        if (typeof result.voice_similarity === "number") {
            if (voiceSimilarityMetric) {
                voiceSimilarityMetric.innerText = (result.voice_similarity * 100).toFixed(1) + "%";
            }
        } else {
            if (voiceSimilarityMetric) {
                voiceSimilarityMetric.innerText = "—";
            }
        }

        updateEmotionResults(result);
        refreshTrainingStatus();

        const translationApplied = Boolean(
            result.translation_applied
            || result?.diagnostics?.text?.translation_applied
        );
        if (translationApplied) {
            showNotification(
                "English input was translated to target language before synthesis.",
                "info"
            );
        }
        
        setupRealAudioPlayer();
        navigateTo("results");
        drawWaveform();
        showNotification("Voice generated successfully", "success");

    } catch (err) {
        showNotification("Error: " + err.message, "error");
        console.error(err);
    } finally {
        showLoading(false);
    }
}

function formatEmotionLabel(emotionValue) {
    const normalized = String(emotionValue || "").trim().toLowerCase();
    if (!normalized) {
        return "Unknown";
    }
    return EMOTION_LABELS[normalized] || (normalized.charAt(0).toUpperCase() + normalized.slice(1));
}

function formatConfidencePercent(confidenceValue) {
    const parsed = Number(confidenceValue);
    if (!Number.isFinite(parsed) || parsed < 0) {
        return "—";
    }
    return `${parsed.toFixed(1)}%`;
}

function buildEmotionSummary(probabilities) {
    if (!probabilities || typeof probabilities !== "object") {
        return "";
    }
    const entries = Object.entries(probabilities)
        .map(([name, score]) => [formatEmotionLabel(name), Number(score)])
        .filter(([, score]) => Number.isFinite(score))
        .sort((a, b) => b[1] - a[1]);

    if (entries.length === 0) {
        return "";
    }

    return entries
        .slice(0, 3)
        .map(([name, score]) => `${name}: ${score.toFixed(1)}%`)
        .join(" | ");
}

function updateEmotionResults(result) {
    console.log("🎭 updateEmotionResults called with:", result);
    
    const emotionMetric = document.getElementById("emotionMetric");
    const predictedEmotionEl = document.getElementById("predictedEmotion");
    const emotionConfidenceEl = document.getElementById("emotionConfidence");
    const probabilitySummaryEl = document.getElementById("emotionProbabilitySummary");
    const correctEmotionSelect = document.getElementById("correctEmotionSelect");

    const predictedEmotion = result?.emotion || result?.predicted_emotion || "unknown";
    const emotionConfidence = result?.emotion_confidence ?? result?.confidence ?? null;
    const probabilities = result?.emotion_probabilities || result?.diagnostics?.emotion?.probabilities;
    const emotionProfile = result?.emotion_profile || result?.diagnostics?.emotion?.profile || null;

    console.log("🎭 Extracted emotion data:", {
        predictedEmotion,
        emotionConfidence,
        probabilities,
        emotionProfile
    });

    const prettyEmotion = formatEmotionLabel(predictedEmotion);
    const confidenceText = formatConfidencePercent(emotionConfidence);

    console.log("🎭 Formatted emotion data:", {
        prettyEmotion,
        confidenceText
    });

    if (emotionMetric) {
        emotionMetric.innerText = confidenceText === "—" ? prettyEmotion : `${prettyEmotion} ${confidenceText}`;
    }
    if (predictedEmotionEl) {
        predictedEmotionEl.innerText = prettyEmotion;
    }
    if (emotionConfidenceEl) {
        emotionConfidenceEl.innerText = confidenceText === "—" ? "Confidence unavailable" : `Confidence ${confidenceText}`;
    }
    if (probabilitySummaryEl) {
        const summary = buildEmotionSummary(probabilities);
        probabilitySummaryEl.innerText = summary || "No probability breakdown available.";
    }

    if (correctEmotionSelect) {
        const normalizedEmotion = String(predictedEmotion || "").trim().toLowerCase();
        if (EMOTION_LABELS[normalizedEmotion]) {
            correctEmotionSelect.value = normalizedEmotion;
        } else {
            correctEmotionSelect.value = "neutral";
        }
    }

    if (
        emotionProfile
        && Number.isFinite(Number(emotionProfile.valence))
        && Number.isFinite(Number(emotionProfile.arousal))
        && Number.isFinite(Number(emotionProfile.dominance))
    ) {
        createEmotionBars(
            Number(emotionProfile.valence),
            Number(emotionProfile.arousal),
            Number(emotionProfile.dominance)
        );
    }
}

async function refreshTrainingStatus() {
    const trainingStatusEl = document.getElementById("trainingStatusText");
    if (!trainingStatusEl) {
        return;
    }

    try {
        const response = await fetch(`${BACKEND_URL}/training-status`);
        if (!response.ok) {
            throw new Error(`status ${response.status}`);
        }
        const status = await response.json();
        if (status?.is_training) {
            trainingStatusEl.innerText = "Background training is currently running.";
        } else {
            trainingStatusEl.innerText = "Training is idle. Feedback will be used in the next training cycle.";
        }
    } catch (error) {
        trainingStatusEl.innerText = "Training status unavailable.";
        console.warn("Failed to refresh training status:", error);
    }
}

async function submitEmotionFeedback() {
    const feedbackBtn = document.getElementById("submitFeedbackBtn");
    const correctEmotionSelect = document.getElementById("correctEmotionSelect");
    const predictedEmotion = String(
        latestSynthesisResult?.emotion
        || latestSynthesisResult?.predicted_emotion
        || ""
    ).trim().toLowerCase();

    if (!predictedEmotion) {
        showNotification("No predicted emotion found for feedback.", "warning");
        return;
    }

    if (!correctEmotionSelect || !correctEmotionSelect.value) {
        showNotification("Select the correct emotion first.", "warning");
        return;
    }

    const feedbackAudio = lastSynthesisReferenceAudio || recordedAudio;
    if (!feedbackAudio) {
        showNotification("No reference audio available for feedback.", "warning");
        return;
    }

    const previousLabel = feedbackBtn ? feedbackBtn.innerHTML : "";
    if (feedbackBtn) {
        feedbackBtn.disabled = true;
        feedbackBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    }

    try {
        const feedbackForm = new FormData();
        feedbackForm.append("correct_emotion", correctEmotionSelect.value);
        feedbackForm.append("predicted_emotion", predictedEmotion);
        feedbackForm.append(
            "audio",
            feedbackAudio,
            lastSynthesisReferenceFilename || feedbackAudio.name || "feedback_reference.wav"
        );

        const response = await fetch(`${BACKEND_URL}/feedback`, {
            method: "POST",
            body: feedbackForm,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload?.error) {
            throw new Error(payload?.error || `Feedback failed with status ${response.status}`);
        }

        if (payload?.training_triggered) {
            showNotification("Feedback saved. Training started in background.", "success");
        } else {
            showNotification("Feedback saved. Thanks, this helps retraining.", "success");
        }
        await refreshTrainingStatus();
    } catch (error) {
        showNotification(`Feedback failed: ${error.message}`, "error");
        console.error("Feedback submission error:", error);
    } finally {
        if (feedbackBtn) {
            feedbackBtn.disabled = false;
            feedbackBtn.innerHTML = previousLabel || '<i class="fas fa-brain"></i> Submit Feedback';
        }
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
// LOADING OVERLAY
// ===========================

function showLoading(show) {

    const overlay = document.getElementById("loadingOverlay");

    overlay.style.display = show ? "flex" : "none";

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
            <h4>Emotion Profile</h4>
            <div class="emotion-bar-container">
                <div class="emotion-bar-item">
                    <span>Valence</span>
                    <div class="emotion-bar-track">
                        <div id="valenceBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="valenceValue">0.00</span>
                </div>
                <div class="emotion-bar-item">
                    <span>Arousal</span>
                    <div class="emotion-bar-track">
                        <div id="arousalBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="arousalValue">0.00</span>
                </div>
                <div class="emotion-bar-item">
                    <span>Dominance</span>
                    <div class="emotion-bar-track">
                        <div id="dominanceBar" class="emotion-bar-fill"></div>
                    </div>
                    <span id="dominanceValue">0.00</span>
                </div>
            </div>
        `;
        metricsSection.appendChild(emotionBars);
    }
    
    // Update emotion bars with real values
    if (valence !== null && arousal !== null && dominance !== null) {
        document.getElementById('valenceBar').style.width = (valence * 100) + '%';
        document.getElementById('valenceValue').innerText = valence.toFixed(2);
        
        document.getElementById('arousalBar').style.width = (arousal * 100) + '%';
        document.getElementById('arousalValue').innerText = arousal.toFixed(2);
        
        document.getElementById('dominanceBar').style.width = (dominance * 100) + '%';
        document.getElementById('dominanceValue').innerText = dominance.toFixed(2);
    }
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

// Clear audio state manually (user-triggered)
function clearAudioState() {
    console.log("🗑️ Manually clearing audio state");
    
    // Clear audio variables
    recordedAudio = null;
    generatedAudio = null;
    activePreviewAudio = null;
    activePreviewUrl = null;
    latestSynthesisResult = null;
    lastSynthesisReferenceAudio = null;
    
    // Clear file input
    const audioFileInput = document.getElementById("audioFileInput");
    if (audioFileInput) {
        audioFileInput.value = "";
    }
    
    // Reset recording status
    const recordingStatus = document.getElementById("recordingStatus");
    if (recordingStatus) {
        recordingStatus.innerText = "Click to start recording";
    }
    
    // Reset emotion displays
    const emotionMetric = document.getElementById("emotionMetric");
    if (emotionMetric) {
        emotionMetric.innerText = "—";
    }
    const predictedEmotionEl = document.getElementById("predictedEmotion");
    if (predictedEmotionEl) {
        predictedEmotionEl.innerText = "—";
    }
    const emotionConfidenceEl = document.getElementById("emotionConfidence");
    if (emotionConfidenceEl) {
        emotionConfidenceEl.innerText = "—";
    }
    const probabilitySummaryEl = document.getElementById("emotionProbabilitySummary");
    if (probabilitySummaryEl) {
        probabilitySummaryEl.innerText = "Waiting for synthesis...";
    }
    
    // Remove any emotion bars
    const emotionBars = document.querySelector('.emotion-bars');
    if (emotionBars) {
        emotionBars.remove();
    }
    
    showNotification("Audio state cleared", "info");
    console.log("✅ Audio state cleared manually");
}

// ===========================
// PAGE INITIALIZATION
// ===========================

// Clear all audio state when page loads
function initializePage() {
    console.log("🔄 Initializing page - clearing audio state");
    
    // Use the same clear logic as manual clear
    clearAudioState();
    
    console.log("✅ Page initialized - audio state cleared");
}

// Initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePage);
} else {
    // DOM already loaded
    initializePage();
}
