//! Prints every hand MediaPipe finds in an image.
//!
//! ```sh
//! ./scripts/fetch-fixtures.sh
//! cargo run --example detect_hands -- models/right_hands.jpg
//! ```

use std::num::NonZeroU32;

use mediapipe::{HandLandmarker, Image, ModelSource, Timestamp};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/right_hands.jpg".into());
    let mut landmarker = HandLandmarker::builder(ModelSource::path("models/hand_landmarker.task"))
        .num_hands(NonZeroU32::new(2).expect("2 should be non-zero"))
        .build()?;
    let image = Image::from_file(&path)?;
    let size = image.size();
    let started = std::time::Instant::now();
    let result = landmarker.detect(&image)?;
    println!(
        "{path}: {}x{}, {} hand(s) in {:?}",
        size.width,
        size.height,
        result.hands.len(),
        started.elapsed()
    );
    for (i, hand) in result.hands.iter().enumerate() {
        let label = hand
            .handedness
            .first()
            .map(|c| format!("{:?} {:.3}", c.category_name, c.score.get()))
            .unwrap_or_default();
        let wrist = hand.landmarks[0].point.to_pixels(size);
        println!(
            "  hand {i}: {label}, {} landmarks, wrist at ({:.1}, {:.1}) px, {} world landmarks",
            hand.landmarks.len(),
            wrist.x(),
            wrist.y(),
            hand.world_landmarks.len()
        );
        if let (Some(w), Some(m)) = (hand.world_landmarks.first(), hand.world_landmarks.get(9)) {
            println!(
                "    wrist to middle knuckle: {:.1} mm",
                w.point.distance_to(m.point) * 1000.0
            );
        }
    }

    // Video mode tracks from the previous frame's landmarks instead of running
    // the palm detector every time, which is what a camera pays per frame.
    let mut video = HandLandmarker::builder(ModelSource::path("models/hand_landmarker.task"))
        .num_hands(NonZeroU32::new(2).expect("2 should be non-zero"))
        .build_for_video()?;
    let mut times: Vec<std::time::Duration> = (0..60)
        .map(|frame| {
            let started = std::time::Instant::now();
            video.detect_for_video(&image, Timestamp::from_millis(frame * 33))?;
            Ok(started.elapsed())
        })
        .collect::<Result<_, mediapipe::Error>>()?;
    times.sort();
    println!(
        "video mode, 60 frames: median {:?}, p95 {:?}",
        times[30], times[57]
    );
    Ok(())
}
