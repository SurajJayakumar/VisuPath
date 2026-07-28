const WebSocket = require('ws');
const wss = new WebSocket.Server({port: 8765, host: '0.0.0.0'});
console.log('[VisuPath] WS server running on port 8765');

const boardClients = new Set();
const ttsClients = new Set();


wss.on('connection', (ws, req) => {
  if (req.url === '/board') {
    boardClients.add(ws);
    console.log('[Board] Arduino connected');
    ws.on('message', (data) => {
      const text = data.toString();
      console.log('[Board] Received:', text);
      ttsClients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(text);
        }
      });
    });
    ws.on('close', () => {
      console.log('[Board] Arduino Disconnected')
      boardClients.delete(ws);
    });
  }

  else if (req.url === '/tts') {
    ttsClients.add(ws);
    console.log('[Text to speech] Browser connected');
    ws.on('close', () => {
      console.log('[Text to speech] Browser disconnected');
      ttsClients.delete(ws);
    });
  }
});