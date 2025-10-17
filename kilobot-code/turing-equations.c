#include "kilolib.h"
#include <stdbool.h>
#include <math.h>

#define MESSAGE_WINDOW 32
#define DT 0.001 // this is minimal dt
#define CLOSE_DISTANCE 75

#define FU 49.373
#define FV -82.998
#define GU 61.178
#define GV -93.445
#define DU 3.26298
#define DV 131.715267

#define CAP(x) ((x) > (65504) ? (65504) : ((x) < -(65504) ? -(65504) : (x)))

uint8_t unique_kilobot_ids[10];
uint8_t unique_count = 0;

float decode_custom_float(uint8_t b1, uint8_t b2);
void encode_custom_float(float value, uint8_t *b1, uint8_t *b2);
void set_msg_encoded_float(float value, uint8_t index);

float u = 0.0f;
float v = 0.0f;
float u_accum = 0;
float v_accum = 0;
float dudt = 0; // for blinks

uint32_t last_reset_time = 0;
uint8_t rand_byte(){
    return rand_soft();
}
float rand_float_range(float min, float max) {
    return min + ((float)rand_byte() / 255.0f) * (max - min);
}
static message_t msg;

void setup() {
    last_reset_time = kilo_ticks;
    float random_value = rand_float_range(-0.1f, 0.1f);
    u = random_value; //random_value;
    v = random_value; //random_value;

    //u += kilo_uid / 100;

    msg.type = NORMAL;
    set_msg_encoded_float(u, 1);  // sets data[1] and data[2]
    set_msg_encoded_float(v, 3);  // sets data[3] and data[4]
    msg.data[0] = kilo_uid;
    msg.crc = message_crc(&msg);
}

void update_u(float add_u) {
    float f = FU * u + FV * v;
    float L = (unique_count > 0) ? (add_u - u * unique_count) : 0.0f;
    dudt = (f + DU * L) * DT;
    u = CAP(u + dudt);
}

void update_v(float add_v) {
    float g = GU * u + GV * v;
    float L = (unique_count > 0) ? (add_v - v * unique_count) : 0.0f;
    float dvdt = (g + DV * L) * DT;
    v = CAP(v + dvdt);
}


void message_rx(message_t *m, distance_measurement_t *d) {
    uint8_t dist = estimate_distance(d);
    if (dist < CLOSE_DISTANCE) {
        uint8_t b1 = m->data[1];
        uint8_t b2 = m->data[2];
        uint8_t b3 = m->data[3];
        uint8_t b4 = m->data[4];


        float recv_u = decode_custom_float(b1, b2);
        float recv_v = decode_custom_float(b3, b4);

        uint8_t sender_id = m->data[0];

        bool is_unique = true;
        for (uint8_t i = 0; i < unique_count; i++) {
            if (unique_kilobot_ids[i] == sender_id) {
                is_unique = false;
                break;
            }
        }

        if (is_unique && unique_count < 10) {
            unique_kilobot_ids[unique_count++] = sender_id;
            u_accum += recv_u;
            v_accum += recv_v;
            //unique_count++;
        }
    }
}

message_t* message_tx() {
    return &msg;
}


// void show_color_for_value(float value) {
//     if (value <= -1)
//         set_color(RGB(0, 0, 1));  // Blue: Very Low
//     else if (value < -0.5)
//         set_color(RGB(0, 1, 1));  // Cyan: Low
//     else if (value <= 0.2)
//         set_color(RGB(0, 1, 0));  // Green: Medium
//     else if (value < 0.5)
//         set_color(RGB(1, 1, 0));  // Yellow: High
//     else
//         set_color(RGB(1, 0, 0));  // Red: Very High
// }

void show_color_for_value(float value) { // new
    if (value <= -6000)
        set_color(RGB(0, 0, 1));  // Blue: Very Low
    else if (value < -4000)
        set_color(RGB(0, 1, 1));  // Cyan: Low
    else if (value < -2000)
        set_color(RGB(1, 0, 1)); // PURPLE
    else if (value <= 2000)
        set_color(RGB(0, 1, 0));  // Green: Medium
    else if (value < 4000)
        set_color(RGB(1, 1, 0));  // Yellow: High
    else
        set_color(RGB(1, 0, 0));  // Red: Very High
}

float absf(float x) {
    return (x < 0) ? -x : x;
}

void loop() {
    if (kilo_ticks > last_reset_time + MESSAGE_WINDOW) {
        last_reset_time = kilo_ticks;

        update_u(u_accum);
        update_v(v_accum);

        show_color_for_value(u);

        if (absf(dudt) > 0.0001f) { // and if (abs(u) > 60000) then (absf(dudt) > 3000f) for flash
        //if ((fabsf(u) > 60000.0f && fabsf(dudt) > 3000.0f) || (fabsf(u) <= 60000.0f && fabsf(dudt) > 0.0001f)) {
            uint8_t blink_ms = 50 + (uint8_t)(absf(dudt) * 100); 
            if (blink_ms > 150) blink_ms = 150;

            if (dudt > 0) {
                set_color(RGB(1, 0, 0));  // Red: Increasing
            } else {
                set_color(RGB(0, 0, 1));  // Blue: Decreasing
            }

            delay(blink_ms);
            show_color_for_value(u);  // Restore base color
        }

        u_accum = 0;
        v_accum = 0;
        unique_count = 0;

        set_msg_encoded_float(u, 1);  // sets data[1] and data[2]
        set_msg_encoded_float(v, 3);  // sets data[3] and data[4]
        msg.crc = message_crc(&msg);
    }

    delay(100);
}

int main() {
    kilo_init();
    kilo_message_tx = message_tx;
    kilo_message_rx = message_rx;
    kilo_start(setup, loop);
    return 0;
}

float decode_custom_float(uint8_t b1, uint8_t b2){ // half float
    uint8_t sign = b1 >> 7;
    uint8_t exponent = (b1 >> 2) & 0x1F; // 5 bits
    uint16_t mantissa = ((b1 & 0x03) << 8) | b2;

    float mant_fl = 0;
    for (int j = 9; j >= 0; j--){
        uint16_t bit = (mantissa >> j) & 1;
        mant_fl += bit * pow(2, j-10);
    }

    if (exponent == 31 && sign == 0) return 65504.0;
    else if (exponent == 31 && sign == 1) return -65504.0;
    else if (exponent == 0 && mantissa == 0) return 0.0;
    else if (exponent == 0) return pow(-1, sign) * pow(2, -14) * mant_fl;
    else return ldexpf(1.0 + mant_fl, exponent - 15) * (sign ? -1.0f : 1.0f);
}

void set_msg_encoded_float(float value, uint8_t index) {
    uint8_t b1, b2;
    encode_custom_float(value, &b1, &b2);
    msg.data[index] = b1;
    msg.data[index + 1] = b2;
}

void encode_custom_float(float value, uint8_t *b1, uint8_t *b2){
    // send +- 65504 with precision 6.1e-5 with 0.001 resolution

    uint8_t sign = (value < 0) ? 1 : 0; // extract sign 
    if (value < 0 ) value = -value; // take absolute
    
    int exponent = 0;
    float normalised = frexp(value, &exponent); // value = normalised * 2^ exponent components
    exponent = exponent + 14; // custom basis of 15
    if (value == 0.0) {
        *b1 = 0;
        *b2 = 0;
        return;
    } else if (value >= 65504.0) {
        *b1 = (sign << 7) | (31 << 2); // Inf
        *b2 = 0;
        return;
    }
    // remove implicit leading 1
    float frac = normalised * 2 - 1.0; // rescale to [0, 1]
    uint16_t mantissa = 0; // convert to 10 bit mantissa
    for (int j = 9; j >= 0; j--) {
        float bit_val = 1.0f / (1 << (10 - j)); // instead of pow(2, j-10)
        if (frac >= bit_val) {
            mantissa |= (1 << j);
            frac -= bit_val;
        }
    }

    *b1 = (sign << 7) | ((exponent & 0x1F) << 2) | ((mantissa >> 8) & 0x03);
    *b2 = mantissa & 0xFF;
}