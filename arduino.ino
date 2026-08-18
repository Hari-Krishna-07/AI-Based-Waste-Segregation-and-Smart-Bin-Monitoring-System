#include <ESP32Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

Servo topServo;
Servo binServo;

LiquidCrystal_I2C lcd(0x27, 16, 2);

// PINS
#define TOP_SERVO_PIN 18
#define BIN_SERVO_PIN 19

#define TRIG_PIN 5
#define ECHO_PIN 21

#define LCD_SDA 22
#define LCD_SCL 23

#define BUZZER_PIN 25

// BIN SETTINGS
#define BIN_HEIGHT 15.0
#define WASTE_LIMIT 7.0


void setup() {

  Serial.begin(115200);

  delay(1000);

  // LCD
  Wire.begin(LCD_SDA, LCD_SCL);

  lcd.init();
  lcd.backlight();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART BIN");

  lcd.setCursor(0, 1);
  lcd.print("Starting...");

  delay(2000);


  // ULTRASONIC
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  digitalWrite(TRIG_PIN, LOW);


  // BUZZER
  pinMode(BUZZER_PIN, OUTPUT);
  noTone(BUZZER_PIN);


  // SERVOS
  topServo.setPeriodHertz(50);
  binServo.setPeriodHertz(50);

  topServo.attach(
    TOP_SERVO_PIN,
    500,
    2500
  );

  binServo.attach(
    BIN_SERVO_PIN,
    500,
    2500
  );


  // START POSITION

  // Top closed
  topServo.write(0);

  // Bin 1
  binServo.writeMicroseconds(500);

  delay(1000);


  // READY

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("SMART BIN");

  lcd.setCursor(0, 1);
  lcd.print("READY");

  Serial.println("SMART BIN READY");
  Serial.println("Waiting for Python...");
}


// =================================================
// ULTRASONIC
// =================================================

float getDistance() {

  digitalWrite(TRIG_PIN, LOW);

  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);

  delayMicroseconds(10);

  digitalWrite(TRIG_PIN, LOW);

  unsigned long duration =
    pulseIn(
      ECHO_PIN,
      HIGH,
      30000
    );

  if (duration == 0) {

    return -1;
  }

  float distance =
    (duration * 0.0343) / 2.0;

  return distance;
}


// =================================================
// CHECK BIN
// =================================================

bool checkBin(int binNumber) {

  float distance =
    getDistance();


  // SENSOR ERROR

  if (distance < 0) {

    Serial.println(
      "ULTRASONIC SENSOR ERROR"
    );

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("BIN ");
    lcd.print(binNumber);

    lcd.setCursor(0, 1);
    lcd.print("SENSOR ERROR");

    tone(
      BUZZER_PIN,
      2000
    );

    delay(500);

    noTone(
      BUZZER_PIN
    );

    return true;
  }


  // WASTE HEIGHT

  float wasteHeight =
    BIN_HEIGHT - distance;


  if (wasteHeight < 0) {

    wasteHeight = 0;
  }


  Serial.print("BIN ");
  Serial.print(binNumber);

  Serial.print(" | Distance: ");
  Serial.print(distance, 1);

  Serial.print(" cm | Waste: ");
  Serial.print(wasteHeight, 1);

  Serial.println(" cm");


  // BIN FULL

  if (wasteHeight > WASTE_LIMIT) {

    Serial.println(
      "BIN FULL"
    );

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("BIN ");
    lcd.print(binNumber);

    lcd.setCursor(0, 1);
    lcd.print("FULL");

    tone(
      BUZZER_PIN,
      2500
    );

    delay(500);

    noTone(
      BUZZER_PIN
    );

    return true;
  }


  // BIN AVAILABLE

  Serial.println(
    "BIN NOT FULL"
  );

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("BIN ");
  lcd.print(binNumber);

  lcd.setCursor(0, 1);
  lcd.print("NOT FULL");

  noTone(
    BUZZER_PIN
  );

  return false;
}


// =================================================
// OPEN TOP
// =================================================

void openTop() {

  Serial.println(
    "OPENING TOP"
  );

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("OPENING TOP");


  for (
    int angle = 0;
    angle <= 90;
    angle++
  ) {

    topServo.write(angle);

    delay(20);
  }


  delay(1000);
}


// =================================================
// CLOSE TOP
// =================================================

void closeTop() {

  Serial.println(
    "CLOSING TOP"
  );

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("CLOSING TOP");


  for (
    int angle = 90;
    angle >= 0;
    angle--
  ) {

    topServo.write(angle);

    delay(20);
  }


  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("TOP CLOSED");
}


// =================================================
// MOVE TO BIN
// =================================================

void moveToBin(
  int binNumber
) {

  Serial.print(
    "MOVING TO BIN "
  );

  Serial.println(
    binNumber
  );


  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print("MOVING TO BIN ");

  lcd.print(
    binNumber
  );


  // BIN 1 - PAPER

  if (
    binNumber == 1
  ) {

    binServo.writeMicroseconds(
      500
    );
  }


  // BIN 2 - PLASTIC

  else if (
    binNumber == 2
  ) {

    binServo.writeMicroseconds(
      1167
    );
  }


  // BIN 3 - METAL

  else if (
    binNumber == 3
  ) {

    binServo.writeMicroseconds(
      1833
    );
  }


  // BIN 4 - GLASS

  else if (
    binNumber == 4
  ) {

    binServo.writeMicroseconds(
      2500
    );
  }


  delay(1500);
}


// =================================================
// PROCESS BIN
// =================================================

void processBin(
  int binNumber,
  String wasteName
) {

  Serial.println();
  Serial.println(
    "=============================="
  );

  Serial.print(
    "WASTE: "
  );

  Serial.println(
    wasteName
  );

  Serial.print(
    "SELECTED BIN: "
  );

  Serial.println(
    binNumber
  );

  Serial.println(
    "=============================="
  );


  // MOVE LOWER SERVO

  moveToBin(
    binNumber
  );

  delay(500);


  // CHECK BIN

  bool full =
    checkBin(
      binNumber
    );


  // BIN FULL

  if (full) {

    Serial.println(
      "BIN FULL - TOP WILL NOT OPEN"
    );

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print(
      wasteName
    );

    lcd.setCursor(0, 1);
    lcd.print(
      "BIN FULL"
    );


    topServo.write(0);


    Serial.println(
      "BIN_FULL"
    );

    delay(2000);

    return;
  }


  // BIN AVAILABLE

  Serial.println(
    "BIN AVAILABLE"
  );

  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(
    wasteName
  );

  lcd.setCursor(0, 1);
  lcd.print(
    "BIN AVAILABLE"
  );

  delay(1000);


  // OPEN TOP

  openTop();

  delay(1500);


  // CLOSE TOP

  closeTop();

  delay(1000);


  Serial.println(
    "SORTING COMPLETE"
  );

  Serial.println(
    "READY"
  );


  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(
    "SORTING DONE"
  );

  lcd.setCursor(0, 1);
  lcd.print(
    wasteName
  );

  delay(2000);


  lcd.clear();

  lcd.setCursor(0, 0);
  lcd.print(
    "SMART BIN"
  );

  lcd.setCursor(0, 1);
  lcd.print(
    "READY"
  );
}


// =================================================
// PROCESS PYTHON COMMAND
// =================================================

void processCommand(
  String command
) {

  command.trim();

  command.toUpperCase();


  Serial.print(
    "RECEIVED: "
  );

  Serial.println(
    command
  );


  // PAPER

  if (
    command == "PAPER"
  ) {

    processBin(
      1,
      "PAPER"
    );
  }


  // PLASTIC

  else if (
    command == "PLASTIC"
  ) {

    processBin(
      2,
      "PLASTIC"
    );
  }


  // METAL

  else if (
    command == "METAL"
  ) {

    processBin(
      3,
      "METAL"
    );
  }


  // GLASS

  else if (
    command == "GLASS"
  ) {

    processBin(
      4,
      "GLASS"
    );
  }


  // TEST

  else if (
    command == "TEST"
  ) {

    Serial.println(
      "TEST COMMAND RECEIVED"
    );

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("TEST OK");

    delay(1000);

    lcd.clear();

    lcd.print("SMART BIN");

    lcd.setCursor(0, 1);
    lcd.print("READY");
  }


  // UNKNOWN

  else {

    Serial.print(
      "UNKNOWN COMMAND: "
    );

    Serial.println(
      command
    );

    lcd.clear();

    lcd.setCursor(0, 0);
    lcd.print("UNKNOWN CMD");

    delay(1000);

    lcd.clear();

    lcd.print("SMART BIN");

    lcd.setCursor(0, 1);
    lcd.print("READY");
  }
}


// =================================================
// MAIN LOOP
// =================================================

void loop() {

  if (
    Serial.available()
  ) {

    String command =
      Serial.readStringUntil(
        '\n'
      );

    processCommand(
      command
    );
  }
}