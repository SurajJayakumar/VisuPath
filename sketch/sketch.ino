#include "Arduino_RouterBridge.h"
#include <Modulino.h>

ModulinoVibro vibro;

void setup() {
  Bridge.begin();
  Monitor.begin();
  Modulino.begin();
  vibro.begin();
  Monitor.println("Modulino Vibro Test Started!");
}

void loop() {
  // --- Test 1: Short buzz (non-blocking) ---
  // Monitor.println("Buzz: SHORT");
  // vibro.on(200, 120);        // ON for 200ms, non-blocking
  // delay(400);
  // vibro.off();
  // delay(500);

  // --- Test 2: Long buzz (blocking — waits automatically) ---
  Monitor.println("Buzz: LONG");
  vibro.on(800, true, MAXIMUM);  // ON for 800ms, blocking (auto-stops)
  delay(500);

   // --- Test 2: Long buzz (blocking — waits automatically) ---
  Monitor.println("Buzz: LONG");
  vibro.on(800, true, 120);  // ON for 800ms, blocking (auto-stops)
  delay(500);

  // --- Test 3: Triple pulse ---
  // Monitor.println("Buzz: TRIPLE PULSE");
  // for (int i = 0; i < 3; i++) {
  //   vibro.on(100, true);   // blocking short pulse
  //   delay(150);
  // }

  delay(2000);
}