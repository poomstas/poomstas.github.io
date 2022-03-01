---
layout: post
title:  "Testing Pump + Fan Simultaneous Use"
date:   2022-02-27
excerpt: "Let's see if a batterpack can power both the pump and the fan simultaneously."
image: "/images/20220301_testing_fan_pump/thumbnail_testing_fan_pump.png"
published: true

---

## Humble Beginnings

As a part of my small-scale smart farm project, I have multiple hardware items that are controlled by a Raspberry Pi. In my current setup, I have a pump and a fan. They are both powered by the battery pack, and are controlled separately by a 2-channel relay.

Because my knowledge in electronics is very limited, I will be taking a very pragmatic, experience-oriented approach to designing my smart farm. The plan is to put together a functional smart farm first, and incrementally improve it as I slowly expand my understanding of relevant subjects like electronics, control systems, plant physiology, and frontend/backend development. It's all for the fun of learning. If you're reading this and see how I could improve, please feel free to let me know :)

Anyways, the circuit diagram looks like below. I used Fritzing to make these diagrams; they are pretty easy to use.

<center><img src="https://github.com/poomstas/poomstas.github.io/blob/master/images/20220301_testing_fan_pump/circuit_diagram.png?raw=true" alt="" style="max-width:100%;" /></center>

For the relay setup, I used a 2-channel relay with a jumper to connect the `JD-VCC` with `VCC`. This effectively allows the relay controls to be powered by the Raspberry Pi. This is ill-advised for the cases where the power going through the relay channels is high because the Raspberry Pi's circuit are not completely isolated, and runs the risk of Raspberry Pi being exposed to the high current going through the relay in case anything goes wrong. For my case, it should be fine because the devices controlled by the relay are powered using a small, low-voltage (<12V) battery pack.

For more information on using relays, I include below a couple of videos that were the most informative.

<iframe width="480" height="360" src="https://www.youtube.com/embed/d9evR-K6FAY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="480" height="360" src="https://www.youtube.com/embed/kfPzXbhTQQk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



The physical configuration looks like below. I've converted a shoe box lid into a test bench for the hardware.

<center><img src="https://github.com/poomstas/poomstas.github.io/blob/master/images/20220301_testing_fan_pump/test_bench.jpg?raw=true" alt="" style="max-width:100%;" /></center>





## Water Pump and Fan Specs

The specs for the water pump and the fan are:

<center><img src="https://github.com/poomstas/poomstas.github.io/blob/master/images/20220301_testing_fan_pump/pump_specs.png?raw=true" alt="" style="max-width:80%;" /></center>

<center><img src="https://github.com/poomstas/poomstas.github.io/blob/master/images/20220301_testing_fan_pump/fan_specs.jpg?raw=true" alt="" style="max-width:80%;" /></center>



And for the battery pack, I used 8 AA Eneloop rechargeable batteries (It has 4 more AA batteries below):

<center><img src="https://github.com/poomstas/poomstas.github.io/blob/master/images/20220301_testing_fan_pump/battery_pack.jpg?raw=true" alt="" style="max-width:80%;" /></center>



I wrote the following code to keep the water pump on, while alternating the relay switch for the fan in 5 second increments. I then used a multimeter to measure the voltage across the water pump to see how much the fan use affects the power delivered to the pump.

```python
'''
Turn on the water pump, and alternate the fan switch in 5 second intervals.
This is to check if the power supplied to the water pump is affected by the fan turning on. 
'''

import time
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM) # Use GPIO Numbers instead of board numbers

# GPIO-7 is No2 (Fan), GPIO-8 is No1 (Water Pump)
GPIO_FAN, GPIO_PUMP = 7, 8
GPIO.setup(GPIO_FAN, GPIO.OUT)      # GPIO Assign mode
GPIO.setup(GPIO_PUMP, GPIO.OUT)     # GPIO Assign mode

GPIO.output(GPIO_FAN, GPIO.HIGH)    # GPIO.HIGH on a relay is OFF
GPIO.output(GPIO_PUMP, GPIO.HIGH)   # GPIO.HIGH on a relay is OFF
time.sleep(1)

try:
    GPIO.output(GPIO_PUMP, GPIO.LOW)

    while True:
        print("Setting No7 to LOW (Fan On)")
        GPIO.output(GPIO_FAN, GPIO.LOW)
        time.sleep(5)

        print("Setting No7 to HIGH (Fan Off)")
        GPIO.output(GPIO_FAN, GPIO.HIGH)
        time.sleep(5)
finally:
    GPIO.cleanup()
```

Measuring the voltages across the water pump, I saw about 0.2V drop when the fans were turning on. The flow rates did not seem to be affected too much, though. 

The results give me confidence that this setup is a viable solution for my mini smart farm project, for now. I will most likely have to consider additional factors when I'm planning to scale up. I will only have to be aware of how the behavior is affected as the battery charges drop over time. 