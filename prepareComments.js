#!/usr/bin/env node

/**
 * prepareComments.js
 * ------------------
 * ACTIVE FILE: This is the main script used by the application's visualization module.
 *
 * Purpose: Processes comments data from the AssemblyAI pipeline for visualization.
 * Function: Reads JSON from stdin, normalizes timestamps, and emits processed data to stdout.
 * Usage: Called by src/visualization/plotly_visualizer.py through the _prepare_comments_js function.
 *
 * Note: Sample comment data for development can be found in src/visualization/sample_comments.js
 */

// Read data from stdin
let data = '';
process.stdin.on('data', (chunk) => {
  data += chunk;
});

process.stdin.on('end', () => {
  try {
    // Parse the input data
    const comments = JSON.parse(data);

    if (!comments || !comments.length) {
      console.error('No comments data provided');
      process.exit(1);
    }

    // Find min and max timestamps to determine the time range
    let minTime = Number.MAX_SAFE_INTEGER;
    let maxTime = 0;

    comments.forEach(comment => {
      const time = comment.create_time || comment.start || 0;
      if (time < minTime) minTime = time;
      if (time > maxTime) maxTime = time;
    });

    // Default to a 60-second range if all timestamps are the same
    if (minTime === maxTime) {
      maxTime = minTime + 60;
    }

    // Use current date as baseline for relative timestamps
    const baseDate = new Date();
    baseDate.setSeconds(0);
    baseDate.setMinutes(0);
    baseDate.setMilliseconds(0);

    // Process the comments
    const processed = comments.map(comment => {
      // Normalize the timestamp
      let createTime = comment.create_time || comment.start || 0;

      // Create a new date object for this comment
      const commentDate = new Date(baseDate);
      // Add the relative seconds from the video
      commentDate.setSeconds(baseDate.getSeconds() + createTime);

      return {
        create_time: createTime,
        text: comment.text || '',
        // Use properly formatted ISO timestamp for visualization
        timestamp: commentDate.toISOString()
      };
    });

    // Sort by time to ensure proper sequencing
    processed.sort((a, b) => a.create_time - b.create_time);

    // Output the processed data
    console.log(JSON.stringify(processed));
  } catch (error) {
    console.error('Error processing comments:', error.message);
    process.exit(1);
  }
});
