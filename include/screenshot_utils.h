#ifndef SCREENSHOT_UTILS_H
#define SCREENSHOT_UTILS_H

#include "common.h"

// Simplified screenshot detection
inline int isLikelyScreenshot(ScreenshotStats stats) {
    // Calculate weighted score
    float score = 0;
    score += stats.edge_score * 0.4;       // Edge characteristics
    score += stats.color_score * 0.3;      // Color uniformity
    score += stats.ui_element_score * 0.3; // UI element indicators
    
    // Output analysis
    printf("Screenshot Analysis:\n");
    printf("Edge Score: %.3f\n", stats.edge_score);
    printf("Color Score: %.3f\n", stats.color_score);
    printf("UI Element Score: %.3f\n", stats.ui_element_score);
    printf("Final Score: %.3f (Threshold: %.1f)\n", score, SCREENSHOT_SCORE_THRESHOLD);
    
    return score > SCREENSHOT_SCORE_THRESHOLD;
}

#endif // SCREENSHOT_UTILS_H 