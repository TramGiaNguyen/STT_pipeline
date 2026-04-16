/**
 * app.js - WebRTC Audio Capture + WebSocket Client cho Streaming STT
 *
 * Chức năng:
 *   1. Lấy âm thanh từ micro qua navigator.mediaDevices.getUserMedia()
 *   2. Gửi audio chunk liên tục qua WebSocket (không lọc client-side VAD)
 *   3. Server xử lý VAD bằng Silero VAD và STT bằng faster-whisper
 *   4. Nhận kết quả text từ server → hiển thị lên UI
 */

// ==================== Cấu hình ====================
const CONFIG = {
    // Audio
    SAMPLE_RATE: 16000,              // 16kHz — chuẩn cho faster-whisper
    CHUNK_DURATION_MS: 500,          // Tăng lên 500ms để đỡ vụn tín hiệu gửi lên
    CHUNK_SIZE: 16000 * 0.5,         // 8000 samples per chunk (500ms @ 16kHz)
    CHANNELS: 1,                     // Mono
    AUDIO_GAIN: 1.0,                 // Không tăng gain (để giữ nguyên chất lượng)

    // Waveform
    WAVEFORM_HISTORY_SEC: 3.0,      // Hiển thị sóng âm trong 3 giây
    WAVEFORM_COLOR_SILENCE: "#334155",
    WAVEFORM_COLOR_SPEECH: "#ef4444",
    WAVEFORM_COLOR_BACKGROUND: "#0f172a",

    // WebSocket
    WS_URL: (() => {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const host = window.location.host || "localhost:8000";
        const params = new URLSearchParams(window.location.search);
        const lang = params.get("lang") || "vi";
        return `${protocol}//${host}/ws/stt?language=${lang}`;
    })(),

    // Debug
    DEBUG_MODE: true,
};

// ==================== Trạng thái ứng dụng ====================
const STATE = {
    isRecording: false,           // Đang thu âm
    wsConnected: false,          // WebSocket đã kết nối
    audioContext: null,          // AudioContext instance
    mediaStream: null,           // MediaStream từ micro
    processor: null,             // AudioWorkletProcessor hoặc ScriptProcessorNode
    ws: null,                    // WebSocket instance

    // Micro selection
    availableMicDevices: [],     // Danh sách micro khả dụng [{deviceId, label}]
    selectedMicId: "",           // ID micro đang chọn
    defaultMicId: "",            // ID micro mặc định

    // Stats
    chunksSent: 0,
    resultsReceived: 0,
    errors: 0,

    // Waveform
    waveformCtx: null,             // Canvas 2D context
    waveformHistory: [],          // Danh sách RMS của các chunk gần đây
    waveformMaxChunks: 0,          // Số chunk tối đa hiển thị (= 3s / 0.256s ≈ 12)
    lastDrawTime: 0,               // Thời gian vẽ cuối cùng (cho animation loop)
};

// ==================== Utility Functions ====================

/**
 * Chuyển AudioBuffer thành PCM 16-bit signed integer array
 */
function audioBufferToPCM16(buffer) {
    const numChannels = buffer.numberOfChannels;
    const length = buffer.length;
    const pcm = new Int16Array(length);

    // Mix tất cả channels về mono
    if (numChannels === 1) {
        for (let i = 0; i < length; i++) {
            let sample = Math.max(-1, Math.min(1, buffer.getChannelData(0)[i]));
            // Áp dụng gain để tăng âm lượng
            sample *= CONFIG.AUDIO_GAIN;
            sample = Math.max(-1, Math.min(1, sample)); // Clamp lại sau khi gain
            pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }
    } else {
        // Mix nhiều channels
        const channels = [];
        for (let c = 0; c < numChannels; c++) {
            channels.push(buffer.getChannelData(c));
        }
        for (let i = 0; i < length; i++) {
            let sum = 0;
            for (let c = 0; c < numChannels; c++) {
                sum += channels[c][i];
            }
            let sample = Math.max(-1, Math.min(1, sum / numChannels));
            // Áp dụng gain
            sample *= CONFIG.AUDIO_GAIN;
            sample = Math.max(-1, Math.min(1, sample)); // Clamp lại
            pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
        }
    }
    return pcm;
}

/**
 * Tính RMS (Root Mean Square) energy của audio buffer — dùng cho WebRTC VAD
 */
function computeRMS(buffer) {
    const data = buffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i] * data[i];
    }
    return Math.sqrt(sum / data.length);
}

/**
 * Chuyển RMS sang decibel (dB)
 */
function rmsToDb(rms) {
    if (rms <= 0) return -100;
    return 20 * Math.log10(rms);
}

/**
 * Mã hóa Int16Array PCM thành base64
 */
function pcmToBase64(pcm) {
    // Tạo ArrayBuffer từ Int16Array
    const buffer = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
    // Chuyển thành Uint8Array
    const uint8 = new Uint8Array(buffer);
    // Mã hóa base64
    let binary = "";
    for (let i = 0; i < uint8.length; i++) {
        binary += String.fromCharCode(uint8[i]);
    }
    return btoa(binary);
}

// ==================== Microphone Selection ====================

/**
 * Lấy danh sách micro khả dụng từ hệ thống
 * Trước tiên yêu cầu quyền truy cập micro để deviceId có label
 */
async function enumerateMicrophones() {
    try {
        // Yêu cầu quyền truy cập micro trước — không có quyền thì deviceId sẽ không có label
        // Dùng fake constraint để xin quyền mà không thực sự thu âm
        const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        tempStream.getTracks().forEach((track) => track.stop());

        // Đợi một chút để devices list cập nhật
        await new Promise((r) => setTimeout(r, 100));

        const devices = await navigator.mediaDevices.enumerateDevices();
        const micDevices = devices
            .filter((d) => d.kind === "audioinput")
            .map((d) => ({
                deviceId: d.deviceId === "" ? "default" : d.deviceId,
                label: d.label || `Micro ${d.deviceId.slice(0, 8)}`,
                groupId: d.groupId,
            }));

        // Loại bỏ trùng lặp (cùng groupId)
        const unique = [];
        const seenGroupIds = new Set();
        for (const mic of micDevices) {
            if (!seenGroupIds.has(mic.groupId)) {
                seenGroupIds.add(mic.groupId);
                unique.push(mic);
            }
        }

        return unique;
    } catch (err) {
        log(`Lỗi liệt kê micro: ${err}`, "error");
        return [];
    }
}

/**
 * Cập nhật dropdown danh sách micro
 */
async function refreshMicrophoneList() {
    const select = document.getElementById("mic-select");
    if (!select) return;

    select.innerHTML = '<option value="">Đang tải...</option>';

    const devices = await enumerateMicrophones();
    STATE.availableMicDevices = devices;

    if (devices.length === 0) {
        select.innerHTML = '<option value="" disabled>Không tìm thấy micro</option>';
        log("Không tìm thấy micro nào trên hệ thống", "warn");
        return;
    }

    // Lưu deviceId mặc định nếu chưa có
    if (!STATE.defaultMicId) {
        STATE.defaultMicId = devices[0].deviceId;
        STATE.selectedMicId = devices[0].deviceId;
    }

    select.innerHTML = "";

    for (const mic of devices) {
        const option = document.createElement("option");
        option.value = mic.deviceId;
        option.textContent = mic.label;
        if (mic.deviceId === STATE.selectedMicId) {
            option.selected = true;
        }
        select.appendChild(option);
    }

    log(`Tìm thấy ${devices.length} micro: ${devices.map((d) => d.label).join(", ")}`, "info");
}

/**
 * Xâyựng constraint cho getUserMedia từ deviceId đã chọn
 */
function buildMediaConstraints(deviceId) {
    return {
        audio: {
            sampleRate: CONFIG.SAMPLE_RATE,
            channelCount: CONFIG.CHANNELS,
            deviceId: deviceId ? { exact: deviceId } : undefined,
            // TẮT sạch bộ lọc tiếng ồn gốc của trình duyệt (giúp Model nhận được tiếng gốc rõ nhất)
            echoCancellation: false,  
            noiseSuppression: false,  
            autoGainControl: false,  
        },
    };
}

// ==================== Audio Processing ====================

/**
 * processAudioChunk - Xử lý mỗi audio chunk từ micro
 *
 * GỬI TẤT CẢ chunk lên server để Silero VAD xử lý
 * Không lọc ở client để server có thể detect silence và kết thúc câu
 */
function processAudioChunk(buffer) {
    const rms = computeRMS(buffer);
    const db = rmsToDb(rms);
    
    if (CONFIG.DEBUG_MODE) {
        console.log(`[Audio] Buffer: ${buffer.length} samples, RMS=${rms.toFixed(6)}, dB=${db.toFixed(1)}`);
    }
    
    // GỬI TẤT CẢ chunk lên server (bao gồm cả silence)
    // Server Silero VAD sẽ quyết định chunk nào là speech
    // Điều này quan trọng để server biết khi nào user dừng nói (silence detection)
    sendChunk(buffer);
    drawWaveformBar(buffer);
}

/**
 * Gửi một chunk audio qua WebSocket ngay lập tức
 */
function sendChunk(buffer) {
    if (!STATE.wsConnected || !STATE.ws) return;
    const pcm16 = audioBufferToPCM16(buffer);
    const b64 = pcmToBase64(pcm16);
    const message = { type: "audio", audio: b64, sample_rate: buffer.sampleRate };
    try {
        STATE.ws.send(JSON.stringify(message));
        STATE.chunksSent++;
        updateStats();
    } catch (e) {
        log(`Lỗi gửi audio: ${e}`, "error");
    }
}

// ==================== WebSocket ====================

/**
 * Kết nối WebSocket tới server
 */
function connectWebSocket() {
    if (STATE.ws && STATE.ws.readyState === WebSocket.OPEN) {
        return;
    }

    log(`Đang kết nối WebSocket: ${CONFIG.WS_URL}`, "info");

    STATE.ws = new WebSocket(CONFIG.WS_URL);

    STATE.ws.onopen = () => {
        STATE.wsConnected = true;
        log("WebSocket đã kết nối", "success");
        updateConnectionStatus();
    };

    STATE.ws.onclose = (event) => {
        STATE.wsConnected = false;
        log(`WebSocket đóng: code=${event.code}, reason=${event.reason || "không có"}`, "warn");
        updateConnectionStatus();
    };

    STATE.ws.onerror = (error) => {
        STATE.errors++;
        log("WebSocket lỗi", "error");
        updateStats();
    };

    STATE.ws.onmessage = (event) => {
        try {
            const msg = JSON.parse(event.data);
            handleServerMessage(msg);
        } catch (e) {
            log(`Lỗi parse message: ${e}`, "error");
        }
    };
}

/**
 * Xử lý message từ server
 */
function handleServerMessage(msg) {
    const type = msg.type || "unknown";

    switch (type) {
        case "status":
            log(`Server status: ${msg.status} — ${msg.message || ""}`, "info");
            updateServerStatus(msg.status, msg.message);
            break;

        case "result":
            STATE.resultsReceived++;
            displayResult(msg.text || "", msg.confidence, msg.timestamp);
            updateStats();
            break;

        case "error":
            STATE.errors++;
            log(`Server error: ${msg.message}`, "error");
            updateStats();
            break;

        case "pong":
            // Ping/pong heartbeat
            break;

        default:
            log(`Unknown message type: ${type}`, "warn");
    }
}

/**
 * Ngắt kết nối WebSocket
 */
function disconnectWebSocket() {
    if (STATE.ws) {
        STATE.ws.close();
        STATE.ws = null;
        STATE.wsConnected = false;
    }
}

// ==================== Audio Capture ====================

/**
 * Bắt đầu thu âm từ micro
 */
async function startRecording() {
    if (STATE.isRecording) {
        log("Đang thu âm rồi", "warn");
        return;
    }

    // Nếu đang chọn micro bị thay đổi khi đang thu âm → restart
    const micSelect = document.getElementById("mic-select");
    const targetMicId = micSelect ? micSelect.value : STATE.selectedMicId;

    try {
        log("Yêu cầu quyền truy cập micro...", "info");

        // Lấy media stream từ micro đã chọn
        STATE.mediaStream = await navigator.mediaDevices.getUserMedia(
            buildMediaConstraints(targetMicId)
        );

        log(`Micro được cấp quyền truy cập: ${targetMicId}`, "success");

        // Tạo AudioContext
        STATE.audioContext = new AudioContext({ sampleRate: CONFIG.SAMPLE_RATE });

        // Tạo MediaStreamSource từ stream
        const source = STATE.audioContext.createMediaStreamSource(STATE.mediaStream);

        // Tạo AudioWorkletNode để xử lý audio (Chrome 66+)
        try {
            await setupAudioWorklet(source);
        } catch (e) {
            log(`AudioWorklet không khả dụng (${e}), fallback sang ScriptProcessor`, "warn");
            setupScriptProcessor(source);
        }

        // Kết nối WebSocket
        connectWebSocket();

        // Khởi tạo waveform
        initWaveform();
        STATE.lastDrawTime = performance.now();
        requestAnimationFrame(waveformAnimationLoop);

        STATE.isRecording = true;
        updateRecordingStatus();

        log("Bắt đầu thu âm...", "success");

    } catch (err) {
        if (err.name === "NotAllowedError") {
            log("Từ chối quyền truy cập micro. Hãy cho phép trong trình duyệt.", "error");
        } else if (err.name === "NotFoundError") {
            log("Không tìm thấy micro. Hãy kiểm tra micro có được kết nối không.", "error");
        } else if (err.name === "OverconstrainedError") {
            log(`Micro '${targetMicId}' không hỗ trợ cấu hình này. Thử micro khác.`, "error");
        } else {
            log(`Lỗi khởi tạo audio: ${err}`, "error");
        }
        throw err;
    }
}

/**
 * Thiết lập AudioWorklet cho xử lý audio (phương thức ưu tiên)
 */
async function setupAudioWorklet(source) {
    // Tạo AudioWorkletProcessor inline — truyền raw Float32Array trực tiếp
    const SAMPLE_RATE = CONFIG.SAMPLE_RATE;
    // 30ms chunk = 480 samples @ 16kHz (real-time tốt hơn)
    const CHUNK_SIZE = Math.floor(SAMPLE_RATE * 0.03);
    const workletCode = `
        class AudioCaptureProcessor extends AudioWorkletProcessor {
            constructor() {
                super();
                this.buffer = new Float32Array(0);
                this.chunkSize = ${CHUNK_SIZE};
            }
            process(inputs, outputs, parameters) {
                const input = inputs[0];
                if (!input || !input[0]) return true;

                // Tích lũy samples cho đến khi đủ CHUNK_SIZE
                const incoming = input[0];
                const newBuf = new Float32Array(this.buffer.length + incoming.length);
                newBuf.set(this.buffer, 0);
                newBuf.set(incoming, this.buffer.length);
                this.buffer = newBuf;

                // Chỉ gửi khi đủ kích thước chunk
                while (this.buffer.length >= this.chunkSize) {
                    const chunk = this.buffer.subarray(0, this.chunkSize);
                    this.buffer = this.buffer.subarray(this.chunkSize);
                    this.port.postMessage({ samples: chunk, sampleRate: ${SAMPLE_RATE} });
                }
                return true;
            }
        }
        registerProcessor("audio-capture-processor", AudioCaptureProcessor);
    `;

    const blob = new Blob([workletCode], { type: "application/javascript" });
    const blobUrl = URL.createObjectURL(blob);

    await STATE.audioContext.audioWorklet.addModule(blobUrl);

    const workletNode = new AudioWorkletNode(
        STATE.audioContext,
        "audio-capture-processor"
    );

    workletNode.port.onmessage = (event) => {
        if (event.data && event.data.samples) {
            const samples = event.data.samples;
            const sampleRate = event.data.sampleRate || SAMPLE_RATE;

            const buffer = new AudioBuffer({
                sampleRate: sampleRate,
                numberOfChannels: 1,
                length: samples.length,
            });
            buffer.getChannelData(0).set(samples);
            processAudioChunk(buffer);
        }
    };

    source.connect(workletNode);
    workletNode.connect(STATE.audioContext.destination);

    STATE.processor = workletNode;
    log("AudioWorklet đã khởi tạo", "success");
}

/**
 * Fallback: thiết lập ScriptProcessorNode (hỗ trợ trình duyệt cũ)
 */
function setupScriptProcessor(source) {
    // ScriptProcessor đã deprecated nhưng vẫn hoạt động
    const bufferSize = Math.floor(CONFIG.SAMPLE_RATE * CONFIG.CHUNK_DURATION_MS / 1000);

    STATE.processor = STATE.audioContext.createScriptProcessor(
        bufferSize,  // frames per callback
        1,           // input channels
        0,           // output channels (không cần output)
    );

    STATE.processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer;
        processAudioChunk(inputData);
    };

    source.connect(STATE.processor);
    STATE.processor.connect(STATE.audioContext.destination);  // Cần kết nối để chạy
}

/**
 * Dừng thu âm
 */
function stopRecording() {
    if (!STATE.isRecording) {
        return;
    }

    STATE.isRecording = false;

    // Dừng processor
    if (STATE.processor) {
        try {
            STATE.processor.disconnect();
        } catch (e) { /* ignore */ }
        STATE.processor = null;
    }

    // Dừng media stream
    if (STATE.mediaStream) {
        STATE.mediaStream.getTracks().forEach((track) => track.stop());
        STATE.mediaStream = null;
    }

    // Đóng AudioContext
    if (STATE.audioContext) {
        STATE.audioContext.close();
        STATE.audioContext = null;
    }

    // Reset waveform
    STATE.waveformHistory = [];
    clearWaveform();

    updateRecordingStatus();
    log("Đã dừng thu âm", "info");
}

// ==================== UI Updates ====================

/**
 * Cập nhật trạng thái nút ghi âm
 */
function updateRecordingStatus() {
    const btnRecord = document.getElementById("btn-record");
    const btnRecordText = document.getElementById("btn-record-text");
    const statusText = document.getElementById("status-text");

    if (STATE.isRecording) {
        if (btnRecordText) btnRecordText.textContent = "Dừng";
        btnRecord.classList.remove("btn-start");
        btnRecord.classList.add("btn-stop");
        statusText.textContent = "Đang nghe...";
        statusText.className = "status-text status-listening";
    } else {
        if (btnRecordText) btnRecordText.textContent = "Bắt đầu";
        btnRecord.classList.remove("btn-stop");
        btnRecord.classList.add("btn-start");
        statusText.textContent = "Sẵn sàng";
        statusText.className = "status-text status-ready";
    }
}

/**
 * Cập nhật trạng thái kết nối WebSocket
 */
function updateConnectionStatus() {
    const connStatus = document.getElementById("conn-status");
    if (!connStatus) return;

    if (STATE.wsConnected) {
        connStatus.textContent = "Đã kết nối";
        connStatus.className = "conn-status connected";
    } else {
        connStatus.textContent = "Chưa kết nối";
        connStatus.className = "conn-status disconnected";
    }
}

/**
 * Cập nhật trạng thái server (từ message server gửi về)
 */
function updateServerStatus(status, message) {
    const statusText = document.getElementById("status-text");
    if (!statusText) return;

    switch (status) {
        case "listening":
            statusText.textContent = message || "Đang nghe...";
            statusText.className = "status-text status-listening";
            break;
        case "processing":
            statusText.textContent = message || "Đang xử lý...";
            statusText.className = "status-text status-processing";
            break;
        case "silence":
            statusText.textContent = message || "Không có lời nói";
            statusText.className = "status-text status-silence";
            break;
        case "error":
            statusText.textContent = message || "Lỗi";
            statusText.className = "status-text status-error";
            break;
    }
}

// ==================== Waveform Visualization ====================

/**
 * Khởi tạo canvas waveform khi bắt đầu thu âm
 */
function initWaveform() {
    const canvas = document.getElementById("waveform-canvas");
    if (!canvas) {
        log("Canvas waveform không tìm thấy!", "error");
        return;
    }

    const ctx = canvas.getContext("2d");
    STATE.waveformCtx = ctx;

    // Tính số chunk tối đa hiển thị (3 giây)
    STATE.waveformMaxChunks = Math.ceil(CONFIG.WAVEFORM_HISTORY_SEC / CONFIG.CHUNK_DURATION_MS * 1000);

    // Đặt kích thước canvas bằng kích thước CSS (force layout trước khi đo)
    const rect = canvas.getBoundingClientRect();

    // Nếu canvas chưa có kích thước, đợi một frame rồi thử lại
    if (rect.width === 0 || rect.height === 0) {
        log(`Canvas chưa có kích thước (${rect.width}x${rect.height}), thử lại sau...`, "warn");
        requestAnimationFrame(() => {
            const rect2 = canvas.getBoundingClientRect();
            canvas.width = Math.floor(rect2.width);
            canvas.height = Math.floor(rect2.height);
            log(`Canvas resize: ${canvas.width}x${canvas.height}`, "info");
            clearWaveform();
        });
    } else {
        canvas.width = Math.floor(rect.width);
        canvas.height = Math.floor(rect.height);
        log(`Canvas khởi tạo: ${canvas.width}x${canvas.height}`, "info");
    }

    // Vẽ nền trống lần đầu
    clearWaveform();

    log("Waveform canvas đã khởi tạo", "info");
}

/**
 * Xóa canvas waveform
 */
function clearWaveform() {
    const ctx = STATE.waveformCtx;
    const canvas = document.getElementById("waveform-canvas");
    if (!ctx || !canvas) return;

    // Nếu canvas chưa có kích thước, thử resize trước
    if (canvas.width === 0 || canvas.height === 0) {
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.floor(rect.width) || 300;
        canvas.height = Math.floor(rect.height) || 80;
    }

    ctx.fillStyle = CONFIG.WAVEFORM_COLOR_BACKGROUND;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Vẽ đường trung tâm
    ctx.strokeStyle = CONFIG.WAVEFORM_COLOR_SILENCE;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();
    ctx.setLineDash([]);
}

/**
 * Vẽ sóng âm waveform
 *
 * Mỗi chunk được biểu diễn bằng một cột thẳng đứng (bar) có chiều cao = RMS energy.
 * Các chunk được vẽ từ trái sang phải theo thứ tự thời gian.
 * Màu thanh thể hiện mức năng lượng: xám nhạt = im lặng, đỏ = lớn.
 */
function drawWaveformBar(buffer) {
    const ctx = STATE.waveformCtx;
    const canvas = document.getElementById("waveform-canvas");
    if (!ctx || !canvas) return;

    // Kiểm tra canvas có kích thước hợp lệ chưa
    if (canvas.width === 0 || canvas.height === 0) {
        // Thử resize lại
        const rect = canvas.getBoundingClientRect();
        canvas.width = Math.floor(rect.width);
        canvas.height = Math.floor(rect.height);
        log(`Canvas resize trong drawWaveformBar: ${canvas.width}x${canvas.height}`, "warn");
        if (canvas.width === 0 || canvas.height === 0) return;
    }

    // Tính RMS energy của chunk
    const rms = computeRMS(buffer);
    const db = rmsToDb(rms);

    // Debug: log dB nếu quá nhỏ hoặc quá lớn
    if (db > -10) {
        console.log(`[Waveform DEBUG] RMS=${rms.toFixed(6)}, dB=${db.toFixed(1)}`);
    }

    // Thêm vào history
    STATE.waveformHistory.push({ rms, db });

    // Giới hạn số chunk trong history
    while (STATE.waveformHistory.length > STATE.waveformMaxChunks) {
        STATE.waveformHistory.shift();
    }

    // Xóa toàn bộ canvas và vẽ lại từ đầu
    drawFullWaveform();
}

/**
 * Vẽ toàn bộ waveform từ history
 */
function drawFullWaveform() {
    const ctx = STATE.waveformCtx;
    const canvas = document.getElementById("waveform-canvas");
    if (!ctx || !canvas) return;

    const w = canvas.width;
    const h = canvas.height;
    if (w === 0 || h === 0) return;

    const centerY = h / 2;
    const totalBars = STATE.waveformHistory.length;

    // Tính bar width để vừa với canvas (bao gồm gap)
    const barGap = 2;
    const availableWidth = w;
    const barWidth = Math.max(3, Math.floor((availableWidth - barGap * totalBars) / totalBars));

    // Xóa nền
    ctx.fillStyle = CONFIG.WAVEFORM_COLOR_BACKGROUND;
    ctx.fillRect(0, 0, w, h);

    // Vẽ đường trung tâm mờ
    ctx.strokeStyle = "rgba(51, 65, 85, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(w, centerY);
    ctx.stroke();

    // dBFS range cho visualization: -70 (rất nhỏ) đến -20 (rất lớn)
    const DB_MIN = -70;
    const DB_MAX = -20;

    for (let i = 0; i < totalBars; i++) {
        const item = STATE.waveformHistory[i];
        const x = i * (barWidth + barGap);

        // Chiều cao thanh: chuẩn hóa từ dB_MIN → dB_MAX ra 0 → 1
        const normalized = Math.max(0, Math.min(1, (item.db - DB_MIN) / (DB_MAX - DB_MIN)));
        const barHeight = Math.max(2, normalized * (h * 0.85));

        // Màu sắc: xám nhạt = im lặng (năng lượng thấp), đỏ = lớn (năng lượng cao)
        const color = normalized > 0.1 ? CONFIG.WAVEFORM_COLOR_SPEECH : CONFIG.WAVEFORM_COLOR_SILENCE;

        // Vẽ thanh đối xứng (trên và dưới đường trung tâm)
        ctx.fillStyle = color;
        ctx.fillRect(x, centerY - barHeight / 2, barWidth, barHeight);

        // Thêm glow effect nhẹ cho thanh năng lượng cao
        if (normalized > 0.25) {
            ctx.shadowColor = CONFIG.WAVEFORM_COLOR_SPEECH;
            ctx.shadowBlur = 3;
            ctx.fillRect(x, centerY - barHeight / 2, barWidth, barHeight);
            ctx.shadowBlur = 0;
        }
    }

    // Vẽ label "MIC ACTIVE" nếu đang thu âm
    if (STATE.isRecording) {
        ctx.fillStyle = "rgba(51, 65, 85, 0.8)";
        ctx.fillRect(w - 90, 4, 86, 18);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "11px JetBrains Mono, monospace";
        ctx.fillText("MIC ACTIVE", w - 86, 17);
    }
}

/**
 * Animation loop cho waveform (requestAnimationFrame)
 * Đảm bảo canvas được vẽ lại đều đặn dù chunk có đến hay không
 */
function waveformAnimationLoop(timestamp) {
    if (!STATE.isRecording) return;

    // Chỉ vẽ lại nếu có chunk mới hoặc đã đủ thời gian (60fps)
    const elapsed = timestamp - STATE.lastDrawTime;
    if (elapsed >= 16) {  // ~60fps
        // Luôn vẽ lại waveform nếu đang thu âm (để xem có chunk mới không)
        drawFullWaveform();
        STATE.lastDrawTime = timestamp;
    }

    requestAnimationFrame(waveformAnimationLoop);
}

/**
 * Hiển thị kết quả nhận dạng lên UI
 */
function displayResult(text, confidence, timestamp) {
    const resultsDiv = document.getElementById("results");
    if (!resultsDiv) return;

    const resultItem = document.createElement("div");
    resultItem.className = "result-item";

    // Thời gian
    const time = timestamp ? new Date(timestamp).toLocaleTimeString("vi-VN") : new Date().toLocaleTimeString("vi-VN");

    // Nội dung
    const confidenceStr = confidence ? ` (${(confidence * 100).toFixed(0)}%)` : "";
    resultItem.innerHTML = `
        <div class="result-time">${time}${confidenceStr}</div>
        <div class="result-text">${text || "<im lặng>"}</div>
    `;

    // Thêm vào đầu danh sách
    resultsDiv.insertBefore(resultItem, resultsDiv.firstChild);

    // Giới hạn số kết quả hiển thị
    const MAX_RESULTS = 50;
    while (resultsDiv.children.length > MAX_RESULTS) {
        resultsDiv.removeChild(resultsDiv.lastChild);
    }
}

/**
 * Cập nhật thống kê
 */
function updateStats() {
    const statsDiv = document.getElementById("stats");
    if (!statsDiv) return;

    statsDiv.innerHTML = `
        Đã gửi: ${STATE.chunksSent} chunks |
        Đã nhận: ${STATE.resultsReceived} kết quả |
        Lỗi: ${STATE.errors}
    `;
}

/**
 * Log message ra console và UI
 */
function log(message, level = "info") {
    console[`${level === "error" ? "error" : level === "warn" ? "warn" : "log"}`](`[STT] ${message}`);

    const logDiv = document.getElementById("log");
    if (!logDiv) return;

    const time = new Date().toLocaleTimeString("vi-VN", { hour12: false });
    const logItem = document.createElement("div");
    logItem.className = `log-item log-${level}`;
    logItem.textContent = `[${time}] ${message}`;

    logDiv.insertBefore(logItem, logDiv.firstChild);

    // Giới hạn số dòng log
    const MAX_LOG = 100;
    while (logDiv.children.length > MAX_LOG) {
        logDiv.removeChild(logDiv.lastChild);
    }
}

// ==================== Event Listeners ====================

/**
 * Khởi tạo khi DOM ready
 */
document.addEventListener("DOMContentLoaded", () => {
    log("Web client khởi tạo", "info");

    // Tải danh sách micro
    refreshMicrophoneList();

    // =============== Tabs Logic ===============
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons and contents
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked button and target content
            button.classList.add('active');
            const targetId = button.getAttribute('data-tab');
            document.getElementById(targetId).classList.add('active');

            // If switching away from realtime tab, stop recording to save resources
            if (targetId !== 'tab-realtime' && STATE.isRecording) {
                stopRecording();
            }
        });
    });

    // =============== File Upload / Stream Logic ===============
    const fileInput        = document.getElementById("file-input");
    const btnUpload        = document.getElementById("btn-upload");
    const btnClearTranscript = document.getElementById("btn-clear-transcript");
    const btnDownload      = document.getElementById("btn-download-transcript");
    const spinner          = document.getElementById("upload-spinner");
    const fileStatusText   = document.getElementById("file-status-text");
    const progressContainer = document.getElementById("file-progress-container");
    const progressFill     = document.getElementById("file-progress-fill");
    const progressLabel    = document.getElementById("file-progress-label");
    const transcriptArea   = document.getElementById("transcript-area");
    const transcriptStats  = document.getElementById("transcript-stats");

    /** Định dạng giây → "mm:ss" hoặc "h:mm:ss" */
    function formatDuration(sec) {
        if (!sec || isNaN(sec)) return "?";
        const h = Math.floor(sec / 3600);
        const m = Math.floor((sec % 3600) / 60);
        const s = Math.floor(sec % 60);
        if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
        return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    }

    /** Cập nhật thanh tiến trình */
    function setProgress(pct, label) {
        if (progressContainer) progressContainer.style.display = "block";
        if (progressFill) progressFill.style.width = `${pct}%`;
        if (progressLabel) progressLabel.textContent = label || `${pct}%`;
    }

    /** Bật/tắt UI trong lúc đang xử lý */
    function setProcessingState(active, fileName) {
        if (btnUpload)  btnUpload.disabled  = active;
        if (fileInput)  fileInput.disabled  = active;
        if (spinner)    spinner.style.display = active ? "inline-block" : "none";
        if (active) {
            fileStatusText.textContent = `Đang xử lý: ${fileName} — vui lòng chờ...`;
            fileStatusText.className   = "status-text status-processing";
        }
    }

    // Khi chọn file
    if (fileInput && btnUpload) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                btnUpload.disabled = false;
                fileStatusText.textContent = `Đã chọn: ${fileInput.files[0].name} (${(fileInput.files[0].size / 1024 / 1024).toFixed(1)} MB)`;
                fileStatusText.className   = "status-text status-listening";
            } else {
                btnUpload.disabled = true;
                fileStatusText.textContent = "Chọn file audio/video để tải lên";
                fileStatusText.className   = "status-text status-ready";
            }
        });

        // ===== Nút "Nhận dạng file" — dùng SSE streaming =====
        btnUpload.addEventListener("click", async () => {
            if (!fileInput.files.length) return;

            const file = fileInput.files[0];
            const params = new URLSearchParams(window.location.search);
            const lang   = params.get("lang") || "vi";

            const formData = new FormData();
            formData.append("file", file);
            formData.append("language", lang);

            const diarizeCheckbox = document.getElementById("diarize-checkbox");
            if (diarizeCheckbox && diarizeCheckbox.checked) {
                formData.append("diarize", "true");
            } else {
                formData.append("diarize", "false");
            }

            // Reset UI
            if (transcriptArea)   { transcriptArea.value = ""; }
            if (btnDownload)      { btnDownload.disabled = true; }
            const btnDownloadSrt = document.getElementById("btn-download-srt");
            if (btnDownloadSrt)   { btnDownloadSrt.disabled = true; }
            if (btnClearTranscript) btnClearTranscript.style.display = "none";
            if (transcriptStats)  transcriptStats.textContent = "";
            if (progressContainer) progressContainer.style.display = "none";

            setProcessingState(true, file.name);
            log(`Bắt đầu stream STT: ${file.name} (${lang})`, "info");

            let totalDuration = 0;
            let segmentCount  = 0;
            let aborted = false;
            let lastSpeakerText = null; // Theo dõi cờ đổi người nói
            // Lưu lại thông tin tất cả segment để xuất file srt/vtt
            window.currentSegments = []; 

            /** Cấu trúc Data lại bằng "Nhịp Thở" (Timestamps) và Bẻ Dòng */
            function smartFormatSegment(eventData) {
                // Nếu model không trả về từng từ (words), dùng text chay từ model
                if (!eventData.words || eventData.words.length === 0) {
                    let s = eventData.text.trim();
                    if (!s) return "";
                    // Đảm bảo viết hoa chữ đầu và kết thúc bằng dấu chấm
                    return s.charAt(0).toUpperCase() + s.slice(1) + (/[.!?]$/.test(s) ? "" : ".");
                }

                let finalSegment = "";
                let wordCountByLine = 0;
                
                // Mảng các từ nối thường dùng ở giữa câu, không nên ngắt dòng trước nó
                const conjunctions = ["mà", "nhưng", "nên", "vì", "tại", "thì", "còn", "vậy", "tuy", "rồi", "nếu", "để"];

                for (let i = 0; i < eventData.words.length; i++) {
                    let wObj = eventData.words[i];
                    let wordStr = wObj.word; // Đã bao gồm dấu câu từ model (lstrip ở server)
                    
                    // 1. Xử lý Viết Hoa: Nếu là từ đầu tiên của segment hoặc sau dấu ngắt câu
                    if (finalSegment.length === 0 || /[.!?]\s*$/.test(finalSegment) || finalSegment.endsWith("\n")) {
                        // Bỏ khoảng trắng thừa ở đầu nếu có
                        wordStr = wordStr.trimStart();
                        wordStr = wordStr.charAt(0).toUpperCase() + wordStr.slice(1);
                    }
                    
                    finalSegment += wordStr;
                    wordCountByLine++;

                    // 2. Kiểm tra dấu câu hiện có từ AI (Faster-Whisper thường gắn dấu vào word)
                    const hasPunctuation = /[.!?,,]/.test(wordStr);

                    // 3. Logic thêm dấu câu dựa trên "Nhịp thở" (Gaps) nếu AI chưa bỏ dấu
                    if (i < eventData.words.length - 1) {
                        let nextWObj = eventData.words[i + 1];
                        let nextWord = nextWObj.word.trim().toLowerCase();
                        let gap = nextWObj.start - wObj.end;
                        
                        // Nếu AI chưa có dấu câu tại đây
                        if (!hasPunctuation) {
                            // Nếu nghỉ rất dài (> 2.0s) => Ngắt đoạn (Paragraph)
                            if (gap >= 2.0) {
                                finalSegment += ".\n\n";
                                wordCountByLine = 0;
                            } 
                            // Nếu nghỉ vừa (> 1.0s) => Kết thúc câu
                            else if (gap >= 1.0) {
                                // Nếu từ tiếp theo là liên từ => chỉ dùng dấu phẩy
                                if (conjunctions.includes(nextWord)) {
                                    finalSegment += ", ";
                                } else {
                                    finalSegment += ". ";
                                }
                                wordCountByLine = 0;
                            } 
                            // Nếu nghỉ ngắn (> 0.4s) => Dấu phẩy
                            else if (gap >= 0.4) {
                                finalSegment += ", ";
                            } 
                            // Nếu câu quá dài (> 15 từ) mà chưa nghỉ, và gặp từ nối => ngắt phẩy nhẹ
                            else if (wordCountByLine >= 15 && conjunctions.includes(nextWord)) {
                                finalSegment += ", ";
                                wordCountByLine = 0;
                            } 
                            else {
                                finalSegment += " ";
                            }
                        } else {
                            // Nếu AI đã có dấu câu, chỉ cần thêm khoảng trắng hoặc xuống dòng nếu là dấu kết thúc
                            if (/[.!?]/.test(wordStr)) {
                                if (gap >= 2.5) {
                                    finalSegment += "\n\n";
                                } else {
                                    finalSegment += " ";
                                }
                                wordCountByLine = 0;
                            } else {
                                // Dấu phẩy hoặc khác
                                finalSegment += " ";
                            }
                        }
                    } else {
                        // Kết thúc segment: Đảm bảo có dấu kết thúc nếu AI quên
                        if (!/[.!?]$/.test(finalSegment)) {
                            finalSegment += ".";
                        }
                    }
                }
                return finalSegment;
            }

            try {
                const response = await fetch("/api/transcribe-stream", {
                    method: "POST",
                    body:   formData,
                });

                if (!response.ok) {
                    const errText = await response.text();
                    throw new Error(`Server lỗi (${response.status}): ${errText}`);
                }

                setProgress(0, "Đang chuyển đổi định dạng âm thanh và lọc nhiễu nền...");

                const reader  = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer    = "";

                // Đọc SSE stream cho đến khi kết thúc
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    // Tách từng SSE event (phân cách bằng \n\n)
                    const parts = buffer.split("\n\n");
                    buffer = parts.pop(); // Giữ phần chưa đầy đủ

                    for (const part of parts) {
                        const line = part.trim();
                        if (!line.startsWith("data:")) continue;

                        let event;
                        try {
                            event = JSON.parse(line.slice(5).trim());
                        } catch {
                            continue;
                        }

                        if (event.type === "info") {
                            // Server gửi thông tin file ngay khi convert xong
                            totalDuration = event.total_duration || 0;
                            const dur = formatDuration(totalDuration);
                            setProgress(0, `Bắt đầu nhận dạng AI... / Tổng: ${dur}`);
                            log(`File duration: ${dur}, ngôn ngữ: ${event.language}`, "info");

                        } else if (event.type === "status") {
                            // Cập nhật trạng thái (ví dụ: "Đang phân tách người nói...")
                            setProgress(0, event.message || "Đang xử lý...");
                            fileStatusText.textContent = event.message || "Đang xử lý...";
                            
                        } else if (event.type === "segment") {
                            // Append text vào textarea
                            segmentCount++;
                            window.currentSegments.push(event); // Lưu dữ liệu gốc

                            if (transcriptArea) {
                                let formattedText = smartFormatSegment(event);
                                
                                // Nếu có phân tách người nói
                                if (event.speaker && event.speaker !== "Không rõ") {
                                    if (event.speaker !== lastSpeakerText) {
                                        // Người nói mới => Xuống dòng và chèn tên
                                        let prefix = `\n\n[${event.speaker}]: `;
                                        if (!transcriptArea.value) prefix = `[${event.speaker}]: `; // Nếu là dòng đầu tiên
                                        
                                        transcriptArea.value += prefix + formattedText;
                                        lastSpeakerText = event.speaker;
                                    } else {
                                        // Cùng người nói => Nối tiếp
                                        transcriptArea.value += (transcriptArea.value && !transcriptArea.value.endsWith("\n") && !transcriptArea.value.endsWith(" ") ? " " : "") + formattedText;
                                    }
                                } else {
                                    // Không có diarization
                                    transcriptArea.value += (transcriptArea.value && !transcriptArea.value.endsWith("\n") && !transcriptArea.value.endsWith(" ") ? " " : "") + formattedText;
                                }
                                
                                // Auto-scroll xuống dưới
                                transcriptArea.scrollTop = transcriptArea.scrollHeight;
                            }
                            // Cập nhật progress
                            const pct = event.progress || 0;
                            const cur = formatDuration(event.end);
                            const tot = formatDuration(totalDuration);
                            setProgress(pct, `${pct}% — ${cur} / ${tot} (${segmentCount} đoạn)`);

                        } else if (event.type === "done") {
                            // Hoàn tất
                            setProgress(100, `✅ Hoàn tất! ${event.total_segments || segmentCount} đoạn — ${formatDuration(totalDuration)}`);
                            log(`Stream STT hoàn tất: ${event.total_segments || segmentCount} segments`, "success");

                        } else if (event.type === "error") {
                            throw new Error(event.message);
                        }
                    }
                }

                // Kết thúc bình thường
                fileStatusText.textContent = `✅ Hoàn tất: ${file.name}`;
                fileStatusText.className   = "status-text status-listening";

                const wordCount = (transcriptArea?.value || "").trim().split(/\s+/).filter(Boolean).length;
                if (transcriptStats) {
                    transcriptStats.textContent = `${segmentCount} đoạn · ${wordCount} từ · ${formatDuration(totalDuration)}`;
                }

                if (transcriptArea?.value.trim()) {
                    if (btnDownload) btnDownload.disabled = false;
                    if (btnDownloadSrt && window.currentSegments.length > 0) btnDownloadSrt.disabled = false;
                }
                if (btnClearTranscript) btnClearTranscript.style.display = "inline-flex";

            } catch (err) {
                if (!aborted) {
                    log(`Lỗi stream STT: ${err.message}`, "error");
                    fileStatusText.textContent = `❌ ${err.message}`;
                    fileStatusText.className   = "status-text status-error";
                    if (progressLabel) progressLabel.textContent = "Đã dừng do lỗi";

                    // Nếu đã có transcript một phần → vẫn cho download
                    if (transcriptArea?.value.trim()) {
                        if (btnDownload) btnDownload.disabled = false;
                        if (btnDownloadSrt && window.currentSegments.length > 0) btnDownloadSrt.disabled = false;
                        if (transcriptStats) transcriptStats.textContent += " (kết quả một phần)";
                    }
                }
            } finally {
                setProcessingState(false, file.name);
                fileInput.value  = ""; // Reset input nhưng giữ transcript
                btnUpload.disabled = true;
            }
        });
    }

    // ===== Hàm tiện ích sinh file SRT =====
    function formatTimeSrt(seconds) {
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
    }

    function generateSrtData(segments) {
        let srtContent = "";
        segments.forEach((seg, index) => {
            srtContent += `${index + 1}\n`;
            srtContent += `${formatTimeSrt(seg.start)} --> ${formatTimeSrt(seg.end)}\n`;
            let text = seg.text;
            if (seg.speaker && seg.speaker !== "Không rõ") {
                text = `[${seg.speaker}]: ${text}`;
            }
            srtContent += `${text}\n\n`;
        });
        return srtContent;
    }

    // ===== Nút Download .txt =====
    if (btnDownload) {
        btnDownload.addEventListener("click", () => {
            const text = transcriptArea?.value?.trim();
            if (!text) return;

            const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href     = url;
            a.download = `transcript_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-")}.txt`;
            a.click();
            URL.revokeObjectURL(url);
            log("Đã tải file transcript .txt", "success");
        });
    }

    // ===== Nút Download .srt =====
    const btnDownloadSrt = document.getElementById("btn-download-srt");
    if (btnDownloadSrt) {
        btnDownloadSrt.addEventListener("click", () => {
            if (!window.currentSegments || window.currentSegments.length === 0) return;

            const srtContent = generateSrtData(window.currentSegments);
            const blob = new Blob([srtContent], { type: "text/plain;charset=utf-8" });
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement("a");
            a.href     = url;
            a.download = `subtitle_${new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-")}.srt`;
            a.click();
            URL.revokeObjectURL(url);
            log("Đã xuất file phụ đề .srt", "success");
        });
    }

    // ===== Nút Xóa transcript =====
    if (btnClearTranscript) {
        btnClearTranscript.addEventListener("click", () => {
            if (transcriptArea) transcriptArea.value = "";
            if (btnDownload)    btnDownload.disabled = true;
            if (transcriptStats) transcriptStats.textContent = "";
            if (progressContainer) progressContainer.style.display = "none";
            fileStatusText.textContent = "Chọn file audio/video để tải lên";
            fileStatusText.className   = "status-text status-ready";
            btnClearTranscript.style.display = "none";
            log("Đã xóa transcript", "info");
        });
    }

    // Lắng nghe sự kiện khi micro được kết nối/ngắt
    navigator.mediaDevices.addEventListener("devicechange", () => {
        log("Phát hiện thay đổi thiết bị, cập nhật danh sách micro...", "info");
        refreshMicrophoneList();
    });

    // Nút refresh danh sách micro
    const btnRefreshMic = document.getElementById("btn-refresh-mic");
    if (btnRefreshMic) {
        btnRefreshMic.addEventListener("click", async () => {
            await refreshMicrophoneList();
        });
    }

    // Khi chọn micro khác
    const micSelect = document.getElementById("mic-select");
    if (micSelect) {
        micSelect.addEventListener("change", (e) => {
            STATE.selectedMicId = e.target.value;
            log(`Đã chọn micro: ${e.target.options[e.target.selectedIndex].text}`, "info");

            // Nếu đang thu âm → restart lại với micro mới
            if (STATE.isRecording) {
                log("Đang restart lại thu âm với micro mới...", "info");
                stopRecording();
                setTimeout(() => startRecording(), 200);
            }
        });
    }

    const btnRecord = document.getElementById("btn-record");
    if (btnRecord) {
        btnRecord.addEventListener("click", async () => {
            if (STATE.isRecording) {
                stopRecording();
            } else {
                try {
                    await startRecording();
                } catch (e) {
                    // Lỗi đã được log trong startRecording
                }
            }
        });
    }

    const btnClear = document.getElementById("btn-clear");
    if (btnClear) {
        btnClear.addEventListener("click", () => {
            const resultsDiv = document.getElementById("results");
            if (resultsDiv) resultsDiv.innerHTML = "";
            log("Đã xóa kết quả", "info");
        });
    }

    // Kết nối WebSocket ngay khi trang load (để xác nhận server online)
    connectWebSocket();


    // Khởi tạo stats
    updateStats();
    updateConnectionStatus();
    updateRecordingStatus();

    // Kiểm tra WebSocket đã kết nối chưa (sẽ kết nối khi bắt đầu thu âm)
    updateConnectionStatus();

    // Xử lý resize canvas
    window.addEventListener("resize", () => {
        const canvas = document.getElementById("waveform-canvas");
        if (canvas) {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            if (STATE.isRecording) {
                drawFullWaveform();
            } else {
                clearWaveform();
            }
        }
    });
});

/**
 * Cleanup khi tab đóng
 */
window.addEventListener("beforeunload", () => {
    stopRecording();
    disconnectWebSocket();
});
