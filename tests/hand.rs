//! Hand landmarker integration tests. Run `./scripts/fetch-fixtures.sh` first;
//! without the fixtures every test that needs them skips rather than fails.
//!
//! The pinned wrist positions were measured through this binding on
//! v0.10.35's CPU delegate. Like the face numbers in `face.rs` they double as
//! an ABI check: a wrong options layout makes `build` fail outright rather
//! than shifting a point.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::mpsc;

use mediapipe::{
    Delegate, Error, Hand, HandLandmarker, Image, IouThreshold, ModelSource, Size, Timestamp,
};

const LANDMARKER: &str = "models/hand_landmarker.task";
const HANDS: &str = "models/right_hands.jpg";

fn fixtures() -> bool {
    let ok = Path::new(LANDMARKER).exists() && Path::new(HANDS).exists();
    if !ok {
        eprintln!("skipping: run ./scripts/fetch-fixtures.sh first");
    }
    ok
}

fn two_hands() -> mediapipe::HandLandmarkerBuilder {
    HandLandmarker::builder(ModelSource::path(LANDMARKER))
        .num_hands(NonZeroU32::new(2).expect("2 should be non-zero"))
}

fn hands_image() -> Image {
    Image::from_file(HANDS).expect("the fetched right_hands.jpg should decode")
}

/// Wrists in pixels, left to right, so the assertion does not depend on the
/// order MediaPipe happens to report hands in.
fn wrists(hands: &[Hand], size: Size) -> Vec<(f32, f32)> {
    let mut wrists: Vec<_> = hands
        .iter()
        .map(|hand| {
            let p = hand.landmarks[0].point.to_pixels(size);
            (p.x(), p.y())
        })
        .collect();
    wrists.sort_by(|a, b| a.0.total_cmp(&b.0));
    wrists
}

#[test]
fn finds_both_hands_at_the_known_positions() {
    if !fixtures() {
        return;
    }
    let image = hands_image();
    let size = image.size();
    assert_eq!(
        size,
        Size {
            width: 720,
            height: 382
        }
    );

    let result = two_hands()
        .build()
        .expect("the fetched hand_landmarker.task should build a landmarker")
        .detect(&image)
        .expect("an image-mode landmarker should accept an image");
    assert_eq!(result.hands.len(), 2, "right_hands.jpg has two hands");

    for hand in &result.hands {
        assert_eq!(hand.landmarks.len(), 21, "the standard hand skeleton");
        assert_eq!(
            hand.world_landmarks.len(),
            21,
            "one world point per landmark"
        );
        let label = hand
            .handedness
            .first()
            .expect("every hand should carry a handedness classification");
        assert_eq!(label.category_name.as_deref(), Some("Right"));
        assert!(
            label.score.get() > 0.9,
            "handedness score {:?}",
            label.score
        );
    }

    let got = wrists(&result.hands, size);
    for ((x, y), (want_x, want_y)) in got.iter().zip([(137.2, 27.6), (582.9, 353.2)]) {
        assert!(
            (x - want_x).abs() < 3.0 && (y - want_y).abs() < 3.0,
            "wrist at ({x}, {y}), expected near ({want_x}, {want_y})"
        );
    }
}

/// World landmarks are metres, not image units: a wrist-to-middle-knuckle
/// length is a few centimetres whatever the image size.
#[test]
fn world_landmarks_are_in_metres() {
    if !fixtures() {
        return;
    }
    let result = two_hands()
        .build()
        .expect("the fetched hand_landmarker.task should build a landmarker")
        .detect(&hands_image())
        .expect("an image-mode landmarker should accept an image");
    for hand in &result.hands {
        let palm = hand.world_landmarks[0]
            .point
            .distance_to(hand.world_landmarks[9].point);
        assert!(
            (0.04..0.15).contains(&palm),
            "wrist to middle knuckle should be a hand's length in metres, got {palm}"
        );
    }
}

#[test]
fn video_mode_tracks_across_frames() {
    if !fixtures() {
        return;
    }
    let mut landmarker = two_hands()
        .build_for_video()
        .expect("the fetched hand_landmarker.task should build a video landmarker");
    let image = hands_image();
    for frame in 0..10 {
        let result = landmarker
            .detect_for_video(&image, Timestamp::from_millis(frame * 33))
            .unwrap_or_else(|e| panic!("frame {frame}: {e}"));
        assert_eq!(result.hands.len(), 2, "frame {frame}");
    }
}

/// The Metal delegate is the one worth using on a Mac — around a tenth of the
/// CPU delegate's time per frame — so it is pinned to find the same hands.
///
/// "The same" within 2% of the image, not to the pixel: the GPU path computes
/// in half precision and lands a few pixels away from the CPU's answer (4px of
/// 382 here). Two hands this image apart cannot be confused at that tolerance.
#[cfg(target_os = "macos")]
#[test]
fn the_metal_delegate_finds_the_same_hands() {
    if !fixtures() {
        return;
    }
    let image = hands_image();
    let size = image.size();
    let cpu = two_hands()
        .build()
        .expect("the CPU delegate should build")
        .detect(&image)
        .expect("CPU detection should succeed");
    let gpu = two_hands()
        .delegate(Delegate::Gpu)
        .build_for_video()
        .expect("the GPU delegate should build on macOS")
        .detect_for_video(&image, Timestamp::from_millis(0))
        .expect("GPU detection should succeed");
    assert_eq!(gpu.hands.len(), cpu.hands.len());
    let tolerance = 0.02 * size.width.max(size.height) as f32;
    for ((gx, gy), (cx, cy)) in wrists(&gpu.hands, size)
        .into_iter()
        .zip(wrists(&cpu.hands, size))
    {
        assert!(
            (gx - cx).hypot(gy - cy) < tolerance,
            "GPU wrist ({gx}, {gy}) should match CPU wrist ({cx}, {cy})"
        );
    }
}

/// As in `face.rs`: assert ordering and delivery, not a frame count.
#[test]
fn live_stream_delivers_results_in_order_and_stops_at_drop() {
    if !fixtures() {
        return;
    }
    let (tx, rx) = mpsc::channel();
    let mut stream = two_hands()
        .build_stream(move |result, _image, timestamp| {
            let hands = result.map(|r| r.hands.len()).unwrap_or(usize::MAX);
            tx.send((timestamp, hands))
                .expect("the receiver should outlive the stream");
        })
        .expect("a free callback slot should be available");

    let image = hands_image();
    let sent: Vec<Timestamp> = (0..10).map(|f| Timestamp::from_millis(f * 33)).collect();
    for (frame, ts) in sent.iter().enumerate() {
        stream
            .send(&image, *ts)
            .unwrap_or_else(|e| panic!("frame {frame}: {e}"));
    }
    drop(stream);

    let got: Vec<_> = rx.try_iter().collect();
    assert!(!got.is_empty(), "no results at all");
    let mut last: Option<Timestamp> = None;
    for (ts, hands) in &got {
        assert!(sent.contains(ts), "unexpected timestamp {ts:?}");
        if let Some(prev) = last {
            assert!(*ts > prev, "timestamps out of order: {prev:?} then {ts:?}");
        }
        last = Some(*ts);
        assert_eq!(*hands, 2, "at {ts:?}");
    }
    assert!(
        matches!(rx.recv(), Err(mpsc::RecvError)),
        "callback outlived the stream"
    );
}

/// `HandAssociationCalculator` `ABSL_CHECK`s its threshold is above zero, so a
/// zero would abort the process inside MediaPipe. It is refused here instead —
/// before the library is even loaded, which is why this needs no fixtures.
#[test]
fn a_zero_tracking_threshold_is_refused_before_reaching_c() {
    let err = HandLandmarker::builder(ModelSource::path(LANDMARKER))
        .min_tracking_confidence(IouThreshold::new(0.0).expect("0 is a valid overlap ratio"))
        .build_for_video()
        .expect_err("a zero tracking threshold should be refused");
    match err {
        Error::OutOfRange {
            quantity, value, ..
        } => {
            assert!(quantity.contains("hand-tracking"), "{quantity}");
            assert_eq!(value, 0.0);
        }
        other => panic!("got {other:?}"),
    }
}
