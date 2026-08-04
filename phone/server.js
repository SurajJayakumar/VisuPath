const { Buffer }  = require('node:buffer');
const WebSocket = require('ws');
const net       = require('net');

const WS_PORT = 8765;
const TTS_SOCK = './tts.sock';
const VLM_SOCK = './vlm.sock';

const MSG_TYPE_OBJECT = 0x01; // YOLO object (binary frame)
const MSG_TYPE_SCENE = 0x02; // VLM snapshot (binary frame)


const wss = new WebSocket.Server({port: WS_PORT, host: '0.0.0.0'});
console.log(`[VisuPath] WS server running on port ${WS_PORT}`);

// TTS Socket

let ttsSock = null;

function connectTTS() {
  ttsSock = net.createConnection(TTS_SOCK);
  ttsSock.on('connect', () => console.log('[TTS] Connected to tts_client'));
  ttsSock.on('error', (e) => console.error(`[TTS] Socket error: ${e.message}`));

  ttsSock.on('close', () => {
    console.log('[TTS] Disconnected: retrying in 2s');
    ttsSock = null;
    setTimeout(connectTTS, 2000);
  });

}

connectTTS();

function sendToTTS(msgType, text) {
  // Frame format: [1-byte type][utf8 text]\n
  // Matches the read_lines + on_line parsing in phone/tts_client.py
  if (ttsSock && ttsSock.writable) {

    const payload = Buffer.from(text, 'utf8');

    const frame   = Buffer.allocUnsafe(1 + payload.length + 1);
    frame[0] = msgType;

    payload.copy(frame, 1);
    frame[frame.length - 1] = 0x0A;  // '\n'

    ttsSock.write(frame);
  } 
  else {
    console.warn('[TTS] Not connected: dropping:', text);
  }
}

// VLM socket

let vlmSock = null;
let vlmBusy = false;
let vlmBuf = '';

function connectVLM() {
  vlmSock = net.createConnection(VLM_SOCK);

  vlmSock.on('connect', () => console.log('[VLM] Connected to VLM runner process'));

  vlmSock.on('data', (chunk) => {

    vlmBuf += chunk.toString('utf8');

    let nl;
    while ((nl = vlmBuf.indexOf('\n')) !== -1){
      
      const line = vlmBuf.slice(0, nl).trim();
      vlmBuf = vlmBuf.slice(nl + 1);

      if (!line){
        continue;
      }

      vlmBusy = false;

      console.log(`[VLM] Scene result: ${line.slice(0, 80)}`);
      sendToTTS(MSG_TYPE_SCENE, line);   // 0x02 type byte — high priority in tts_client
    }
  });

  vlmSock.on('error', (e) => console.error(`[VLM] Socket error: ${e.message}`));

  vlmSock.on('close', () => {

    console.log('[VLM] Disconnected: retrying in 2 seconds.');
    vlmSock = null;
    vlmBusy = false;
    vlmBuf = '';
    setTimeout(connectVLM, 2000);
  });

}

connectVLM();


function sendToVLM(base64payload) {

  if (!vlmSock || !vlmSock.writable) {

    console.warn('[VLM] Not connected: dropping snapshot');
    return;
  }

  if (vlmBusy) {

    console.log('[VLM] Busy: dropping snapshot (inference in flight)');
    return;
  }

  vlmBusy = true;
  vlmSock.write(base64payload + '\n');
  console.log('[VLM] Snapshot dispatched');

}

// Websocket server

let boardWs = null; // assuming one board connection at a time 


wss.on('connection', (ws, req) => {

  if (req.url === '/board') {

    if (boardWs !== null) {
      console.warn('[Board] Second connection rejected. Only 1 board allowed');
      ws.close();
      return;
    }

    boardWs = ws;

    console.log('[Board] Arduino connected');

    ws.on('message', (data) => {
      
      // first byte in every frame is MSG TYPE (eg: OBJECT detction text or  base64 image for scene detection), rest of bytes are payload
      if (!Buffer.isBuffer(data) || data.length < 2) {
        console.warn('[Board] Malformed frame, ignoring');
        return;
      }

      const msgType = data[0];
      const payload = data.slice(1).toString('utf8');

      if (msgType === MSG_TYPE_OBJECT) {
        console.log('[Board] Alert:', payload);
        sendToTTS(MSG_TYPE_OBJECT, payload);
      }

      else if (msgType === MSG_TYPE_SCENE) {
        sendToVLM(payload);
      }

      else {
        console.warn('[Board] Unknown message type:', msgType);
      }
    });

    ws.on('close', () => {
      console.log('[Board] Arduino Disconnected');
      boardWs = null;
    });

    ws.on('error', (e) => {
      console.error('[Board] WS error:', e.message);
      boardWs = null;
    });
  }
});