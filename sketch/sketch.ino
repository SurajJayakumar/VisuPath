#include "Arduino_RouterBridge.h"
#include <Modulino.h>

ModulinoVibro vibro;
ModulinoDistance distance;

const float TOO_CLOSE_MM = 700.0;
const float CRITICAL_MM = 300.0;
const unsigned long READ_INTERVAL_MS = 100;
const unsigned long ALERT_INTERVAL_MS = 1000;

unsigned long lastReadAt = 0;
unsigned long lastAlertAt = 0;
bool wasTooClose = false;

int vibrationIntensity(float distanceMm) {
  if (distanceMm <= CRITICAL_MM) {
    return MAXIMUM;
  }

  if (distanceMm >= TOO_CLOSE_MM) {
    return 0;
  }

  float closeness = (TOO_CLOSE_MM - distanceMm) / (TOO_CLOSE_MM - CRITICAL_MM);
  return 120 + int(closeness * (MAXIMUM - 120));
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  Modulino.begin();
  vibro.begin();
  distance.begin();
  Monitor.println("VisuPath distance sensing started.");
}

void loop() {
  unsigned long now = millis();
  if (now - lastReadAt < READ_INTERVAL_MS) {
    return;
  }
  lastReadAt = now;

  float distanceMm = distance.get();
  if (distanceMm <= 0) {
    return;
  }

  bool isTooClose = distanceMm <= TOO_CLOSE_MM;
  if (!isTooClose) {
    if (wasTooClose) {
      Monitor.println("CLEAR: obstacle no longer too close");
    }
    wasTooClose = false;
    vibro.off();
    return;
  }

  int intensity = vibrationIntensity(distanceMm);
  vibro.on(120, false, intensity);

  if (!wasTooClose || now - lastAlertAt >= ALERT_INTERVAL_MS) {
    Monitor.print("TOO_CLOSE: obstacle ahead, ");
    Monitor.print(distanceMm);
    Monitor.println(" mm away");
    lastAlertAt = now;
  }

  wasTooClose = true;
}
