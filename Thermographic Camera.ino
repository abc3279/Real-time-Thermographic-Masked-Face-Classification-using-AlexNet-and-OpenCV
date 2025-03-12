#include <Wire.h>
#include <Adafruit_AMG88xx.h>

Adafruit_AMG88xx amg;

float pixels[AMG88xx_PIXEL_ARRAY_SIZE];
float high_tmp = 0;

void setup() 
{
    Serial.begin(9600);

    bool status;
    
    // default settings
    status = amg.begin();
    if (!status) 
    {
        Serial.print("Could not find a valid AMG88xx sensor, check wiring!");
        // while (1);
    }
    
    delay(100); // let sensor boot up
}


void loop() { 
  amg.readPixels(pixels);
  for(int i=0; i<=AMG88xx_PIXEL_ARRAY_SIZE; i++)
  {
    if(pixels[i] < pixels[i+1])
    {
      high_tmp = pixels[i+1];
    }     
  }
  Serial.println(high_tmp);
  delay(145);
}
