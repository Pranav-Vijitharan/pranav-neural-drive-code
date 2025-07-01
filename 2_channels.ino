// Dual Channel EMG Filter for ESP32
// Channels: GPIO 34 and GPIO 35
// Output: ch1_filtered,ch1_envelope,ch2_filtered,ch2_envelope

// Pin definitions
#define CH1_PIN 34  // EXG Pill 1 (Jaw/Masseter)
#define CH2_PIN 35  // EXG Pill 2 (Chin/Submental)

// Sampling parameters
#define SAMPLE_RATE 500
#define BAUD_RATE 115200

// Envelope buffer parameters
#define BUFFER_SIZE 64

// Channel 1 variables
int ch1_circular_buffer[BUFFER_SIZE];
int ch1_data_index = 0;
int ch1_sum = 0;

// Channel 2 variables
int ch2_circular_buffer[BUFFER_SIZE];
int ch2_data_index = 0;
int ch2_sum = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  
  // Configure ADC
  analogReadResolution(12);  // 12-bit resolution (0-4095)
  analogSetAttenuation(ADC_11db);  // Full 3.3V range
  
  // Initialize buffers
  for(int i = 0; i < BUFFER_SIZE; i++) {
    ch1_circular_buffer[i] = 0;
    ch2_circular_buffer[i] = 0;
  }
  
  delay(2000);  // Wait for serial connection
}

void loop() {
  // Calculate elapsed time for precise timing
  static unsigned long past = 0;
  unsigned long present = micros();
  unsigned long interval = present - past;
  past = present;

  // Run timer
  static long timer = 0;
  timer -= interval;

  // Sample and process both channels
  if(timer < 0) {
    timer += 1000000 / SAMPLE_RATE;  // Reset timer for next sample
    
    // === CHANNEL 1 PROCESSING ===
    // Read raw EMG
    int ch1_sensor_value = analogRead(CH1_PIN);
    
    // Apply EMG filter
    int ch1_filtered = EMGFilter_Ch1(ch1_sensor_value);
    
    // Get envelope
    int ch1_envelope = getEnvelope_Ch1(abs(ch1_filtered));
    
    // === CHANNEL 2 PROCESSING ===
    // Read raw EMG
    int ch2_sensor_value = analogRead(CH2_PIN);
    
    // Apply EMG filter
    int ch2_filtered = EMGFilter_Ch2(ch2_sensor_value);
    
    // Get envelope
    int ch2_envelope = getEnvelope_Ch2(abs(ch2_filtered));
    
    // Output: ch1_filtered,ch1_envelope,ch2_filtered,ch2_envelope
    Serial.print(ch1_filtered);
    Serial.print(",");
    Serial.print(ch1_envelope);
    Serial.print(",");
    Serial.print(ch2_filtered);
    Serial.print(",");
    Serial.println(ch2_envelope);
  }
}

// === CHANNEL 1 FUNCTIONS ===

// Envelope detection for Channel 1
int getEnvelope_Ch1(int abs_emg) {
  ch1_sum -= ch1_circular_buffer[ch1_data_index];
  ch1_sum += abs_emg;
  ch1_circular_buffer[ch1_data_index] = abs_emg;
  ch1_data_index = (ch1_data_index + 1) % BUFFER_SIZE;
  return (ch1_sum / BUFFER_SIZE) * 2;
}

// EMG Bandpass Filter for Channel 1 (74.5-149.5 Hz)
float EMGFilter_Ch1(float input) {
  float output = input;
  {
    static float z1, z2; // filter section state
    float x = output - 0.05159732*z1 - 0.36347401*z2;
    output = 0.01856301*x + 0.03712602*z1 + 0.01856301*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -0.53945795*z1 - 0.39764934*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - 0.47319594*z1 - 0.70744137*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.00211112*z1 - 0.74520226*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  return output;
}

// === CHANNEL 2 FUNCTIONS ===

// Envelope detection for Channel 2
int getEnvelope_Ch2(int abs_emg) {
  ch2_sum -= ch2_circular_buffer[ch2_data_index];
  ch2_sum += abs_emg;
  ch2_circular_buffer[ch2_data_index] = abs_emg;
  ch2_data_index = (ch2_data_index + 1) % BUFFER_SIZE;
  return (ch2_sum / BUFFER_SIZE) * 2;
}

// EMG Bandpass Filter for Channel 2 (74.5-149.5 Hz)
float EMGFilter_Ch2(float input) {
  float output = input;
  {
    static float z1, z2; // filter section state
    float x = output - 0.05159732*z1 - 0.36347401*z2;
    output = 0.01856301*x + 0.03712602*z1 + 0.01856301*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -0.53945795*z1 - 0.39764934*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - 0.47319594*z1 - 0.70744137*z2;
    output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  {
    static float z1, z2; // filter section state
    float x = output - -1.00211112*z1 - 0.74520226*z2;
    output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
    z2 = z1;
    z1 = x;
  }
  return output;
}