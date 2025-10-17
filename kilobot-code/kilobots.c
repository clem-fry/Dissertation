#include "kilolib.h"
#include <stdbool.h>

#define MESSAGE_WINDOW 32
#define SCALE 100L
#define DT 1L  // Scaled version of 0.001

#define FU (445L * SCALE)
#define FV (-533L * SCALE)
#define GU (956L * SCALE)
#define GV (-986L * SCALE)
#define DU (24L * SCALE)
#define DV (114L * SCALE)

#define CAP(x) ((x) > (10*SCALE) ? (10*SCALE) : ((x) < -(10*SCALE) ? -(10*SCALE) : (x)))

int32_t u = SCALE;
int32_t v = SCALE;
int32_t u_accum = 0;
int32_t v_accum = 0;
uint8_t message_count = 0;
int32_t dudt = 0; // for blinks

uint32_t last_reset_time = 0;
message_t msg;

uint8_t unique_kilobot_ids[10];
uint8_t unique_count = 0;

void setup() {
    last_reset_time = kilo_ticks;
    u += kilo_uid / 2;

    msg.type = NORMAL;
    msg.data[0] = kilo_uid;
    msg.data[1] = (uint8_t)((u + SCALE) / 8);
    msg.data[2] = (uint8_t)((v + SCALE) / 8);
    msg.crc = message_crc(&msg);
}

void update_u(int32_t add_u) {
    int64_t f = (int64_t)FU * u + (int64_t)FV * v;
    int32_t L = add_u - u * message_count;
    dudt = f + (int64_t)DU * L;
    dudt = dudt / (SCALE*DT);
    u = CAP(u + (int32_t)dudt);
}

void update_v(int32_t add_v) {
    int64_t g = (int64_t)GU * u + (int64_t)GV * v;
    int32_t L = add_v - v * message_count;
    int64_t dvdt = g + (int64_t)DV * L;
    dvdt = dvdt / (SCALE*DT);
    v = CAP(v + (int32_t)dvdt);
}

void message_rx(message_t *m, distance_measurement_t *d) {
    int32_t recv_u = ((int32_t)m->data[1]) * 8 - SCALE;
    int32_t recv_v = ((int32_t)m->data[2]) * 8 - SCALE;
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
        message_count++;
    }
}

message_t* message_tx() {
    return &msg;
}

void message_tx_success() {
    msg.data[0] = kilo_uid;
    msg.data[1] = (uint8_t)((u + SCALE) / 8);
    msg.data[2] = (uint8_t)((v + SCALE) / 8);
    msg.crc = message_crc(&msg);
}

void show_color_for_value(int32_t value) {
    if (value <= -SCALE)
        set_color(RGB(0, 0, 1));  // Blue: Very Low
    else if (value < SCALE * 0.25)
        set_color(RGB(0, 1, 1));  // Cyan: Low
    else if (value <= SCALE * 1.25)
        set_color(RGB(0, 1, 0));  // Green: Near SCALE
    else if (value < SCALE * 1.75)
        set_color(RGB(1, 1, 0));  // Yellow: High
    else
        set_color(RGB(1, 0, 0));  // Red: Very High
}

int32_t abs32(int32_t x) {
    return (x < 0) ? -x : x;
}

void loop() {
    if (kilo_ticks > last_reset_time + MESSAGE_WINDOW) {
        last_reset_time = kilo_ticks;

        update_u(u_accum);
        update_v(v_accum);

        show_color_for_value(u);

        // Only blink if change is significant (threshold = 2% of SCALE)
        if (abs32(dudt) > SCALE / 50) {
            // Blink duration proportional to |dudt|
            uint8_t blink_ms = 50 + (uint8_t)(abs32(dudt) / (SCALE / 20)); // 50–100 ms

            // Cap the blink time to something reasonable
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
        message_count = 0;
        unique_count = 0;

        msg.data[1] = (uint8_t)((u + SCALE) / 8);
        msg.data[2] = (uint8_t)((v + SCALE) / 8);
        msg.crc = message_crc(&msg);
    }

    delay(100);
}

int main() {
    kilo_init();
    kilo_message_tx = message_tx;
    kilo_message_tx_success = message_tx_success;
    kilo_message_rx = message_rx;
    kilo_start(setup, loop);
    return 0;
}
