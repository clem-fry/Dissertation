// #include "kilolib.h"

// #define MAX_NEIGHBORS 10
// #define MESSAGE_WINDOW 64  // ~10 seconds (320 ticks)
// #define CLOSE_DISTANCE 100
// #define MIN_MESSAGES_FOR_NEIGHBOR 2
// #define TARGET_ID 368 // !!!!
 
// typedef struct {
//     uint8_t id;
//     uint8_t count;
// } neighbor_t;

// neighbor_t neighbors[MAX_NEIGHBORS];
// uint8_t num_neighbors = 0;
// uint8_t message_count = 0;

// message_t msg;

// uint32_t last_reset_time = 0;

// // ----------------------
// // Neighbor tracking logic
// // ----------------------

// int8_t find_neighbor(uint8_t id) {
//     for (uint8_t i = 0; i < num_neighbors; i++) {
//         if (neighbors[i].id == id) {
//             return i;
//         }
//     }
//     return -1;
// }

// void update_neighbor(uint8_t id) {
//     int8_t index = find_neighbor(id);
//     if (index != -1) {
//         neighbors[index].count++;
//     } else if (num_neighbors < MAX_NEIGHBORS) {
//         neighbors[num_neighbors].id = id;
//         neighbors[num_neighbors].count = 1;
//         num_neighbors++;
//     }
// }

// uint8_t count_confirmed_neighbors() {
//     uint8_t count = 0;
//     for (uint8_t i = 0; i < num_neighbors; i++) {
//         if (neighbors[i].count >= MIN_MESSAGES_FOR_NEIGHBOR) {
//             count++;
//         }
//     }
//     return count;
// }

// void reset_neighbors() {
//     num_neighbors = 0;
//     for (uint8_t i = 0; i < MAX_NEIGHBORS; i++) {
//         neighbors[i].id = 0;
//         neighbors[i].count = 0;
//     }
// }

// // -----------------
// // LED color display
// // -----------------

// void update_color(uint8_t r, uint8_t g, uint8_t b) {
//     set_color(RGB(r, g, b));
// }

// void update_color_based_on_density(uint8_t count) {
//     if (message_count < 1)
//         update_color(0, 0, 1);      // Blue
//     else if (message_count == 1)
//         update_color(0, 1, 1);      // Cyan
//     else if (message_count == 2)
//         update_color(0, 1, 0);      // Green
//     else if (count == 3)
//         update_color(1, 1, 0);      // Yellow
//     else
//         update_color(1, 0, 0);      // Red
// }

// // -------------
// // Kilobot setup
// // -------------

// void setup() {
//     msg.type = NORMAL;
//     msg.data[0] = kilo_uid;
//     msg.crc = message_crc(&msg);
//     last_reset_time = kilo_ticks;
// }

// void message_rx(message_t *m, distance_measurement_t *d) {
//     if (m->data[0] == TARGET_ID) {
//         uint8_t sender_id = m->data[0];
//         uint8_t dist = estimate_distance(d);
//         message_count++;

//         if (dist < CLOSE_DISTANCE ) {
//             update_neighbor(sender_id);
//         }
//     }
// }

// message_t *message_tx() {
//     return &msg;
// }

// void message_tx_success() {
//     // Optional: flash or log
// }

// // ------------------
// // Main behavior loop
// // ------------------

// void loop() {
//     uint8_t confirmed = count_confirmed_neighbors();
//     // Reset every MESSAGE_WINDOW (~10s)
//     if (kilo_ticks - last_reset_time > MESSAGE_WINDOW) {
//         update_color_based_on_density(confirmed);
//         last_reset_time = kilo_ticks;
//         reset_neighbors();
//         message_count = 0;
//     }

//     delay(100);
// }

// // -------------------
// // Main entry point
// // -------------------

// int main() {
//     kilo_init();
//     kilo_message_tx = message_tx;
//     kilo_message_tx_success = message_tx_success;
//     kilo_message_rx = message_rx;
//     kilo_start(setup, loop);
//     return 0;
// }

#include "kilolib.h"

#define MESSAGE_WINDOW 64         // ~10 seconds
#define MAX_UNIQUE_SENDERS 20     // Max number of unique IDs to track
#define DISTANCE_THRESHOLD 65    // Only count messages closer than this

message_t msg;

uint8_t unique_ids[MAX_UNIQUE_SENDERS];
uint8_t num_unique = 0;
uint32_t last_reset_time = 0;

// ----------------------
// Check if ID is already seen
// ----------------------

uint8_t is_new_sender(uint8_t id) {
    for (uint8_t i = 0; i < num_unique; i++) {
        if (unique_ids[i] == id) {
            return 0;  // Already seen
        }
    }
    return 1;  // New sender
}

// ----------------------
// LED color logic
// ----------------------

void update_color_based_on_count(uint8_t count) {
    if (count < 4)
        set_color(RGB(0, 0, 1)); // Blue
    else if (count < 6)
        set_color(RGB(1, 0, 1)); // Pink
    else if (count == 6)
        set_color(RGB(1, 1, 1)); // white
    else if (count == 7)
        set_color(RGB(0, 1, 1)); // Cyan
    else if (count == 8)
        set_color(RGB(0, 1, 0)); // Green
    else if (count == 9)
        set_color(RGB(1, 1, 0)); // Yellow
    else
        set_color(RGB(1, 0, 0)); // Red
}

// ----------------------
// Setup, Message Send/Receive
// ----------------------

void setup() {
    msg.type = NORMAL;
    msg.data[0] = kilo_uid;
    msg.crc = message_crc(&msg);

    num_unique = 0;
    last_reset_time = kilo_ticks;
    set_color(RGB(0, 0, 0));  // Start black
}

void message_rx(message_t *m, distance_measurement_t *d) {
    uint8_t sender_id = m->data[0];
    uint8_t distance = estimate_distance(d);

    if (distance < DISTANCE_THRESHOLD && is_new_sender(sender_id)) {
        if (num_unique < MAX_UNIQUE_SENDERS) {
            unique_ids[num_unique] = sender_id;
            num_unique++;
        }
    }
}

message_t *message_tx() {
    return &msg;
}

void message_tx_success() {
    // Optional
}

// ----------------------
// Main Loop
// ----------------------

void loop() {
    if (kilo_ticks - last_reset_time > MESSAGE_WINDOW) {
        update_color_based_on_count(num_unique);
        num_unique = 0;
        last_reset_time = kilo_ticks;
    }
}

// ----------------------
// Entry Point
// ----------------------

int main() {
    kilo_init();
    kilo_message_tx = message_tx;
    kilo_message_tx_success = message_tx_success;
    kilo_message_rx = message_rx;
    kilo_start(setup, loop);
    return 0;
}
