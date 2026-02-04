// ---------------- TRAFFIC LIGHT CONTROLLER (NO SENSORS) ----------------

// ---------- TRAFFIC LIGHT SETUP ----------
const int NUM_ROADS = 4;

// Road 1..4: green pins
const int greenPins[NUM_ROADS] = {3, 5, 7, 9};
// Road 1..4: red pins
const int redPins[NUM_ROADS]   = {4, 6, 8, 10};

// Times in milliseconds
const unsigned long GREEN_NORMAL   = 15000UL;  // 15 seconds
const unsigned long GREEN_PRIORITY = 30000UL;  // 30 seconds
const unsigned long ALL_RED_TIME   = 30000UL;  // 30 seconds


// ---------- MODES ----------
enum Mode {
  MODE_NORMAL = 0,
  MODE_R1_PRIORITY,
  MODE_R2_PRIORITY,
  MODE_R3_PRIORITY,
  MODE_R4_PRIORITY,
  MODE_ALL_RED
};

// ---------- SYSTEM STATE MACHINE ----------
enum SystemState {
  STATE_WAIT_MODE,
  STATE_RUNNING_SEQ,
  STATE_EMERGENCY_ALL_RED
};

Mode currentMode = MODE_NORMAL;
int currentRoad = 0;
unsigned long lastChangeTime = 0;

SystemState systemState = STATE_WAIT_MODE;
unsigned long allRedStart = 0;

// ---------- FUNCTION DECLARATIONS ----------
void setLightsForRoad(int road);
void setAllRed();
unsigned long getGreenTimeForRoad(int road);
void advanceToNextRoad();
void handleSerialInput();
void startEmergencyAllRed();

// ---------- SETUP ----------
void setup() {
  Serial.begin(115200);
  delay(2000);

  for (int i = 0; i < NUM_ROADS; i++) {
    pinMode(greenPins[i], OUTPUT);
    pinMode(redPins[i], OUTPUT);
  }

  setAllRed();
  systemState = STATE_WAIT_MODE;
  currentMode = MODE_NORMAL;
  currentRoad = 0;
  lastChangeTime = millis();

  Serial.println("READY");
}

// ---------- MAIN LOOP ----------
void loop() {
  unsigned long now = millis();

  handleSerialInput();

  switch (systemState) {

    case STATE_EMERGENCY_ALL_RED:
      if (now - allRedStart >= ALL_RED_TIME) {
        systemState = STATE_WAIT_MODE;
        setAllRed();
        Serial.println("ALL_RED_DONE:WAITING_FOR_MODE");
      }
      break;

    case STATE_WAIT_MODE:
      break;

    case STATE_RUNNING_SEQ: {
      unsigned long greenTime = getGreenTimeForRoad(currentRoad);
      if (now - lastChangeTime >= greenTime) {
        advanceToNextRoad();
      }
      break;
    }
  }
}

// ---------- TRAFFIC LIGHT FUNCTIONS ----------
void setLightsForRoad(int road) {
  for (int i = 0; i < NUM_ROADS; i++) {
    if (i == road) {
      digitalWrite(greenPins[i], HIGH);
      digitalWrite(redPins[i], LOW);
    } else {
      digitalWrite(greenPins[i], LOW);
      digitalWrite(redPins[i], HIGH);
    }
  }
}

void setAllRed() {
  for (int i = 0; i < NUM_ROADS; i++) {
    digitalWrite(greenPins[i], LOW);
    digitalWrite(redPins[i], HIGH);
  }
}

unsigned long getGreenTimeForRoad(int road) {
  switch (currentMode) {
    case MODE_NORMAL:
      return GREEN_NORMAL;

    case MODE_R1_PRIORITY:
      return (road == 0) ? GREEN_PRIORITY : GREEN_NORMAL;

    case MODE_R2_PRIORITY:
      return (road == 1) ? GREEN_PRIORITY : GREEN_NORMAL;

    case MODE_R3_PRIORITY:
      return (road == 2) ? GREEN_PRIORITY : GREEN_NORMAL;

    case MODE_R4_PRIORITY:
      return (road == 3) ? GREEN_PRIORITY : GREEN_NORMAL;

    default:
      return GREEN_NORMAL;
  }
}

// ---------- MODIFIED PART ----------
void advanceToNextRoad() {
  // Loop continuously — NO all-red between cycles
  currentRoad++;

  if (currentRoad >= NUM_ROADS) {
    currentRoad = 0;  // restart cycle
    Serial.println("CYCLE_RESTARTED");
  }

  setLightsForRoad(currentRoad);
  lastChangeTime = millis();
}

// ---------- EMERGENCY ----------
void startEmergencyAllRed() {
  systemState = STATE_EMERGENCY_ALL_RED;
  allRedStart = millis();
  currentMode = MODE_ALL_RED;
  setAllRed();
  Serial.println("EMERGENCY_ALL_RED_STARTED:30s");
}

// ---------- SERIAL ----------
void handleSerialInput() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c < '0' || c > '5') continue;

    if (c == '5') {
      startEmergencyAllRed();
      return;
    }

    int modeNum = c - '0';

    if (systemState == STATE_RUNNING_SEQ ||
        systemState == STATE_EMERGENCY_ALL_RED) {
      Serial.print("MODE_IGNORED:");
      Serial.println(modeNum);
      return;
    }

    Mode newMode = MODE_NORMAL;
    switch (modeNum) {
      case 0: newMode = MODE_NORMAL; break;
      case 1: newMode = MODE_R1_PRIORITY; break;
      case 2: newMode = MODE_R2_PRIORITY; break;
      case 3: newMode = MODE_R3_PRIORITY; break;
      case 4: newMode = MODE_R4_PRIORITY; break;
    }

    currentMode = newMode;
    currentRoad = 0;
    lastChangeTime = millis();
    setLightsForRoad(currentRoad);
    systemState = STATE_RUNNING_SEQ;

    Serial.print("MODE_STARTED:");
    Serial.println(modeNum);
  }
}
