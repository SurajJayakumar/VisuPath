#include "Arduino_RouterBridge.h"
#include <Modulino.h>

ModulinoVibro vibro;
ModulinoDistance distance;

const float TOO_FAR_MM = 1100.0;
const float TOO_CLOSE_MM = 300.0;

const unsigned long LOG_INTERVAL_MS = 500;
unsigned long lastLogAt = 0;


int SUPERMAX_VIBRATION = 120; //  calling it supermax because MAXIMUM is already defined as 50.

int vibrationIntensity(float distanceMm) {

   if (distanceMm >= TOO_FAR_MM) { // far way, no vibration
    return 0;
  }

  else if (distanceMm <= TOO_CLOSE_MM) { // too close, vibrate hard
    return SUPERMAX_VIBRATION;
  }


  float closeness = (TOO_FAR_MM - distanceMm) / (TOO_FAR_MM - TOO_CLOSE_MM);
  return GENTLE + int(closeness * (SUPERMAX_VIBRATION - GENTLE));
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

  if (!distance.available()) {
    // bad read, keep current state, no flickering
    return;
  }

  // read distance every single loop
  float distanceMm = distance.get();

  int intensity = vibrationIntensity(distanceMm);

  if (intensity == 0) {
    // immediate stop, object is far / gone
    vibro.off();
  }
  else {
    // Nonblocking, continuous, updates every loop
    // 50 ms duration keeps it alive without blocking
    vibro.on(50, false, intensity);
  }

  // Slow down serial logging only
  unsigned long now = millis();

  if (now - lastLogAt >= LOG_INTERVAL_MS) {
    
    Monitor.print("Distance: ");
    Monitor.print(distanceMm);
    Monitor.print(" mm | Intensity: ");
    Monitor.println(intensity);
    lastLogAt = now;
  }

}
